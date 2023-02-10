import numpy as np
import os

from ibp.model import QNNBoundPropModel
import tensorflow as tf

import argparse
from quantized_layers import QuantizedModel

parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=4, type=int)
parser.add_argument("--timeout", default=0, type=int)
parser.add_argument("--cont", default=0, type=int)
parser.add_argument("--model", default="small")
parser.add_argument("--n", default=0, type=int)
args = parser.parse_args()


@tf.function
def flatten_item(x, y):
    x = tf.reshape(x, [32 * 32 * 3])
    return x, y


batch_size = 256
cifar = tf.keras.datasets.cifar10
_, (x_test, y_test) = cifar.load_data()
x_test = x_test.reshape([-1, 32, 32, 3])
x_test = x_test.astype(np.float32)
if args.n > 0:
    x_test = x_test[: args.n]
    y_test = y_test[: args.n]
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(flatten_item)
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

model = QNNBoundPropModel(ff_model)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    robustify_samples="correct_only",
    robustness_input_min_max=(0, 255),
    regularize_internal_bounds=True,
)
model.restore_checkpoint(f"weights/cifar10_{args.model}_ckpt-{args.cont}")
os.makedirs("results", exist_ok=True)

standard_acc, robust_acc, timeouts = model.evaluate_robustness(
    valid_dataset,
    args.eps,
    total_samples=x_test.shape[0],
    per_sample_timeout=args.timeout,
)
with open(f"results/cifar10_alg.txt", "a") as f:
    f.write(
        f"model={args.model} (cont={args.cont}, n={args.n}), eps={args.eps} alg_timeout={args.timeout}\n"
    )
    f.write(f"standard_acc: {100*standard_acc:0.2f}%\n")
    f.write(f"robust_acc: {100*robust_acc:0.2f}%\n")
    f.write(f"timeouts: {100*timeouts:0.2f}%\n\n")