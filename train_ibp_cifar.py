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


@tf.function
def cutout(x: tf.Tensor, h: int, w: int) -> tf.Tensor:
    img_h = int(x.shape[0])
    img_w = int(x.shape[1])
    c = int(x.shape[2])
    x0 = tf.random.uniform([], 0, img_h + 1 - h, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, img_w + 1 - w, dtype=tf.int32)

    zeros = tf.constant(np.zeros([h, w, c]), dtype=tf.float32)
    mask = tf.pad(
        zeros, [[x0, img_h - x0 - h], [y0, img_w - y0 - w], [0, 0]], constant_values=1.0
    )
    return tf.cast(x, tf.float32) * mask


@tf.function
def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(x, 32 + 8, 32 + 8)
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    x = cutout(x, 8, 8)
    return x, y


@tf.function
def flatten_item(x, y):
    x = tf.reshape(x, [32 * 32 * 3])
    return x, y


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--eps", default=4, type=int)
parser.add_argument("--cont", default=0, type=int)
parser.add_argument("--model", default="small")
parser.add_argument("--wd", default=0, type=float)

args = parser.parse_args()


batch_size = 2048
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train = x_train.reshape([-1, 32, 32, 3])
x_test = x_test.reshape([-1, 32, 32, 3])
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=60000)
    .repeat()
    .map(augmentation)
    .map(flatten_item)
    .batch(batch_size)
)
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(flatten_item)
# train_steps = x_train.shape[0] // batch_size
train_steps = 2000
test_steps = x_test.shape[0] // batch_size

int_bits_activation = 2
int_bits_bias = 2
if args.model == "cnn":
    layers = [(64, 5, 2), (96, 3, 1), (128, 3, 2), 128, 10]  # reduced 256->128
    int_bits_activation = 3
    int_bits_bias = 4
elif args.model == "cnn2":
    layers = [(64, 5, 2), (96, 3, 1), (128, 3, 2), 128, 10]  # reduced 256->128
    int_bits_activation = 3
    int_bits_bias = 4
elif args.model == "cnn3":
    layers = [(64, 5, 2), (96, 3, 1), (128, 3, 2), 128, 10]  # reduced 256->128
    int_bits_activation = 3
    int_bits_bias = 4
elif args.model == "mid":
    layers = [(48, 5, 2), (64, 3, 1), (64, 3, 2), 128, 10]  # reduced 256->128
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
elif args.model == "small":
    layers = [(48, 5, 2), (64, 3, 2), 128, 10]
else:
    raise ValueError(f"Unknown model {args.model}")

ff_model = QuantizedModel(
    layers=layers,
    input_bits=8,
    quantization_bits=8,
    input_reshape=(32, 32, 3),
    int_bits_activation=int_bits_activation,
    int_bits_bias=int_bits_bias,
    last_layer_signed=True,
    last_layer_no_act=True,
)

ff_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ],
)
ff_model(tf.zeros((1, 32 * 32 * 3)))
ff_model.summary()

if args.cont == 0:
    # First train the non-quantized model
    real_model = ff_model.get_non_quantized_copy()
    real_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ],
    )
    real_model(tf.zeros((1, 32 * 32 * 3)))

    real_model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        epochs=20,
        validation_data=valid_dataset.batch(512),
        validation_freq=1,
    )
    ff_model.copy_from_non_quantized(real_model)

    # Then train the quantized model
    ff_model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        epochs=20,
        validation_data=valid_dataset.batch(512),
        validation_freq=1,
    )


model = QNNBoundPropModel(ff_model)
model.compile(
    optimizer=tfa.optimizers.AdamW(weight_decay=args.wd, learning_rate=0.0001),
    # robustify_samples="correct_only",
    robustify_samples="all",
    robustness_input_min_max=(0, 255),
    regularize_internal_bounds=True,
)
# model.restore_checkpoint("weights/cifar_cnn_ckpt3-1")
outfile = f"weights/cifar10_{args.model}_ckpt-{args.cont}"

start_time = time.time()
if args.cont > 0:
    model.restore_checkpoint(f"weights/cifar10_{args.model}_ckpt-{args.cont-1}")
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

model.fit(
    train_dataset,
    epochs=args.epochs,
    steps_per_epoch=train_steps,
    valid_data=valid_dataset.batch(512),
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
# ff_model.save_weights("weights/cifar_cnn.h5")

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
# for i in list(range(1, args.eps + 1)) + [0.00001, 0]:
#     standard_acc, robust_acc, timeouts = model.evaluate_robustness(
#         valid_dataset, i, total_samples=x_test.shape[0], per_sample_timeout=0
#     )
#     print(f"Robustness results on test set with eps = {i}:")
#     print(f"   standard_acc= {100*standard_acc:0.4f}")
#     print(f"   robust_acc  = {100*robust_acc:0.4f}")
#     print(f"   timeouts    = {100*timeouts:0.4f}")

standard_acc, robust_acc, timeouts = model.evaluate_robustness(
    valid_dataset, args.eps, total_samples=x_test.shape[0], per_sample_timeout=0
)
_, robust_acc1, _ = model.evaluate_robustness(
    valid_dataset, 1, total_samples=x_test.shape[0], per_sample_timeout=0
)
with open(f"results/cifar10.txt", "a") as f:
    cmd_line = " ".join(sys.argv[1:])
    f.write(f"python3 train_ibp_cifar.py {cmd_line}\n")
    f.write(f"std_accuracy: {standard_acc*100:0.3f}%\n")
    f.write(f"robust_acc: {robust_acc*100:0.3f}% @{args.eps}\n")
    f.write(f"robust_acc: {robust_acc1*100:0.3f}% @{1}\n")
    f.write(f"timeouts: {timeouts*100:0.3f}%\n\n")