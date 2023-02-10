import sys

import numpy as np
import tensorflow as tf
import os
from ibp.model import QNNBoundPropModel
from ibp.training_schedules import LinearLossFactorSchedule
from tqdm import tqdm
import argparse
import time
from quantized_layers import QuantizedModel
import tensorflow_addons as tfa


class LambdaSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_value, step_fn):
        self.initial_value = initial_value
        self.step_fn = step_fn

    def __call__(self, step):
        return self.initial_value * self.step_fn(step)


@tf.function
def augmentation_mnist(x, y):
    x = tf.image.resize_with_crop_or_pad(x, 28 + 8, 28 + 8)
    x = tf.image.random_crop(x, [28, 28, 1])
    # x = tf.image.random_flip_left_right(x)
    return x, y


@tf.function
def augmentation_fashion(x, y):
    x = tf.image.resize_with_crop_or_pad(x, 28 + 8, 28 + 8)
    x = tf.image.random_crop(x, [28, 28, 1])
    x = tf.image.random_flip_left_right(x)
    return x, y


@tf.function
def flatten_item(x, y):
    x = tf.reshape(x, [28 * 28 * 1])
    return x, y


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--model", default="cnn")
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--eps", default=4, type=int)
parser.add_argument("--cont", default=0, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--wd", default=0, type=float)
args = parser.parse_args()

batch_size = 512
int_bits_activation = 2
int_bits_bias = 2
if args.model == "cnn":
    layers = [(64, 5, 2), (96, 3, 2), 128, 10]  # reduced 256->128
    int_bits_activation = 3
    int_bits_bias = 3
elif args.model == "deep":
    layers = [(64, 5, 2), (96, 3, 1), (128, 3, 2), 128, 10]  # reduced 256->128
    int_bits_activation = 3
    int_bits_bias = 5
elif args.model == "d2":
    layers = [(64, 5, 2), (96, 3, 1), (128, 3, 2), 128, 10]  # reduced 256->128
    int_bits_activation = 2
    int_bits_bias = 4
elif args.model == "bigmnist":
    layers = [
        (64, 5, 2),
        (128, 3, 1),
        (256, 3, 1),
        (384, 3, 1),
        (512, 3, 2),
        128,
        10,
    ]
    int_bits_activation = 2
    int_bits_bias = 4
elif args.model == "big":
    layers = [
        (64, 5, 2),
        (128, 3, 1),
        (256, 3, 1),
        (384, 3, 1),
        (512, 3, 2),
        128,
        10,
    ]  # reduced 256->128
    int_bits_activation = 3
    int_bits_bias = 4
elif args.model == "mid":
    layers = [(64, 5, 2), (96, 3, 2), 128, 10]  # reduced 256->128
elif args.model == "small":
    layers = [(48, 5, 2), (64, 3, 2), 128, 10]
else:
    raise ValueError(f"Unknown model {args.model}")
if args.dataset == "mnist":
    mnist = tf.keras.datasets.mnist
    aug_fn = augmentation_mnist
elif args.dataset == "fashion":
    mnist = tf.keras.datasets.fashion_mnist
    aug_fn = augmentation_fashion
else:
    raise ValueError(f"Unknown dataset '{args.dataset}'")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape([-1, 28, 28, 1])
x_test = x_test.reshape([-1, 28, 28, 1])
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=60000)
    .repeat()
    .map(aug_fn)
    .map(flatten_item)
    .batch(batch_size)
)
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(flatten_item)
test_steps = x_test.shape[0] // batch_size
# train_steps = x_train.shape[0] // batch_size
train_steps = 5000

ff_model = model = QuantizedModel(
    layers,
    input_bits=8,
    quantization_bits=8,
    input_reshape=(28, 28, 1),
    last_layer_signed=True,
    last_layer_no_act=True,
    int_bits_activation=int_bits_activation,
    int_bits_bias=int_bits_bias,
)

ff_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ],
)
ff_model(tf.zeros((1, 28 * 28)))
ff_model.summary()

assert len(tf.config.list_physical_devices("GPU")) > 0
ff_model.fit(
    train_dataset,
    steps_per_epoch=train_steps,
    epochs=1,
    validation_data=valid_dataset.batch(512),
    validation_freq=1,
)

outfile = f"weights/{args.dataset}_{args.model}_ckpt-{args.cont}"

decay_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    # [100000, 150000, 200000], [1.0, 0.25, 0.25**2,0.25**3]
    [
        int(args.epochs / 3) * train_steps,
        2 * int(args.epochs / 3) * train_steps,
    ],
    [1.0, 0.3, 0.1],
)
learning_rate_fn = LambdaSchedule(args.lr, decay_fn)
weight_decay_fn = LambdaSchedule(args.wd, decay_fn)
model = QNNBoundPropModel(ff_model)
model.compile(
    # optimizer=tf.keras.optimizers.Adam(0.0001),
    optimizer=tfa.optimizers.AdamW(
        weight_decay=weight_decay_fn, learning_rate=learning_rate_fn
    ),
    robustify_samples="correct_only",
    robustness_input_min_max=(0, 255),
    regularize_internal_bounds=True,
)

if args.cont > 0:
    model.restore_checkpoint(f"weights/{args.dataset}_{args.model}_ckpt-{args.cont-1}")
    robustness_epsilon = args.eps + 0.1
    robustness_bound_margin = 0.25
    robustness_loss_factor = LinearLossFactorSchedule(
        0.5, 2.0, int(0.9 * train_steps * args.epochs)
    )
    regularization_factor = LinearLossFactorSchedule(
        0.001, 0.005, int(0.9 * train_steps * args.epochs)
    )
    if args.cont == 2:
        robustness_loss_factor = LinearLossFactorSchedule(
            2.0, 5.0, int(0.9 * train_steps * args.epochs)
        )
        regularization_factor = LinearLossFactorSchedule(
            0.005, 0.01, int(0.9 * train_steps * args.epochs)
        )
else:
    robustness_epsilon = LinearLossFactorSchedule(
        0.0, args.eps + 0.1, total_train_steps=int(0.9 * train_steps * args.epochs)
    )
    robustness_bound_margin = 0.2
    robustness_loss_factor = LinearLossFactorSchedule(
        0.0, 0.5, int(0.9 * train_steps * args.epochs)
    )
    regularization_factor = LinearLossFactorSchedule(
        0.0, 0.001, int(0.9 * train_steps * args.epochs)
    )

start_time = time.time()
model.fit(
    train_dataset,
    epochs=args.epochs,
    steps_per_epoch=train_steps,
    valid_data=valid_dataset.batch(2048),
    attack_method=None,
    verbose_schedule=True,
    robustness_epsilon=robustness_epsilon,
    robustness_bound_margin=robustness_bound_margin,
    robustness_loss_factor=robustness_loss_factor,
    regularization_factor=regularization_factor,
)

os.makedirs("weights", exist_ok=True)
os.makedirs("results", exist_ok=True)
model.write_checkpoint(outfile)

train_logs = model.evaluate(train_dataset, steps=train_steps)
print("train_logs: ", str(train_logs))
valid_logs = model.evaluate(valid_dataset.batch(512))
print(
    "Train robust loss: {:0.4}, train acc: {:0.2f}%".format(
        train_logs["robust_loss"], 100 * train_logs["accuracy"]
    )
)
print(
    "Valid robust loss: {:0.4}, valid acc: {:0.2f}%".format(
        valid_logs["robust_loss"], 100 * valid_logs["accuracy"]
    )
)
standard_acc, robust_acc, timeouts = model.evaluate_robustness(
    valid_dataset, args.eps, total_samples=x_test.shape[0], per_sample_timeout=0
)
os.makedirs("results", exist_ok=True)
with open(f"results/{args.dataset}.txt", "a") as f:
    cmd_line = " ".join(sys.argv[1:])
    f.write(f"python3 train_ibp_mnist.py {cmd_line}\n")
    f.write(f"std_accuracy: {standard_acc*100:0.3f}%\n")
    f.write(f"robust_acc: {robust_acc*100:0.3f}% @{args.eps}\n")
    f.write(f"timeouts: {timeouts*100:0.3f}%\n\n")

for i in range(1, args.eps + 1):
    standard_acc, robust_acc, timeouts = model.evaluate_robustness(
        valid_dataset, i, total_samples=x_test.shape[0], per_sample_timeout=0
    )
    print(f"dataset = {args.dataset}, eps={i}, alg_timeout=0")
    print(f"standard_acc: {100*standard_acc:0.2f}%")
    print(f"robust_acc: {100*robust_acc:0.2f}%")
    print(f"timeouts: {100*timeouts:0.2f}%\n")