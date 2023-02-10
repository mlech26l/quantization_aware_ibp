import numpy as np
import tensorflow as tf


def quantize_int(float_value, num_bits, frac_bits):
    # max and min values expressable with int_n
    min_value, max_value = int_get_min_max(num_bits, frac_bits)
    float_value = np.clip(float_value, min_value, max_value)

    scaled = float_value * (2 ** frac_bits)
    quant = np.int32(scaled)
    # Rounding:
    if type(quant) == np.ndarray:
        # Vectorized
        incs = (scaled - quant) >= 0.5
        decs = (scaled - quant) <= -0.5
        quant[incs] += 1
        quant[decs] -= 1
    else:
        # Scalar
        if scaled - quant >= 0.5:
            quant += 1
        elif scaled - quant <= -0.5:
            quant -= 1

    return np.int32(quant)


def quantize_uint(float_value, num_bits, frac_bits):
    # max and min values expressable with uint_n
    min_value, max_value = uint_get_min_max(num_bits, frac_bits)
    float_value = np.clip(float_value, min_value, max_value)

    scaled = float_value * (2 ** frac_bits)
    quant = np.int32(scaled)
    # Rounding:
    if type(quant) == np.ndarray:
        # Vectorized
        incs = (scaled - quant) >= 0.5
        decs = (scaled - quant) <= -0.5
        quant[incs] += 1
        quant[decs] -= 1
    else:
        # Scalar
        # print("scaled: ", str(scaled))
        # print("quant: ", str(quant))
        # print("rem: ", str(scaled - quant))
        # print("rem: {:.12f}".format(scaled - quant))
        if scaled - quant >= 0.5:
            # print("hello")
            quant += 1
        elif scaled - quant <= -0.5:
            # print("bello")
            # print("v: ", str(scaled - quant))
            quant -= 1

    return np.uint32(quant)


def de_quantize_uint(int_value, num_bits, frac_bits):
    real = np.float32(int_value)
    real = real / (2 ** frac_bits)
    return real


def de_quantize_int(int_value, num_bits, frac_bits):
    real = np.float32(int_value)
    real = real / (2 ** frac_bits)
    return real


def get_activation_eps(quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_weights"]
    eps = 0.2 / (2 ** frac_bits)
    return eps


def quantize_weight(v, quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_weights"]
    return quantize_int(v, num_bits, frac_bits)


def quantize_bias(v, quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_bias"]
    return quantize_int(v, num_bits, frac_bits)


def binary_str_to_int(binary_str):
    value = 0
    twos_complement = False
    if binary_str[0] == "1":
        twos_complement = True

    for i in range(1, len(binary_str)):
        value *= 2
        if (not twos_complement and binary_str[i] == "1") or (
            twos_complement and binary_str[i] == "0"
        ):
            value += 1

    if twos_complement:
        value += 1
        value = -value
    return value


def binary_str_to_uint(binary_str):
    value = 0

    for i in range(len(binary_str)):
        value *= 2
        if binary_str[i] == "1":
            value += 1
    return value


def uint_get_min_max_integer(num_bits, frac_bits):
    min_value = 0
    max_value = 2 ** num_bits - 1
    return (min_value, max_value)


def int_get_min_max_integer(num_bits, frac_bits):
    # Need to subtract one bit for the sign bit
    num_value_bits = num_bits - 1
    min_value = -(2 ** num_value_bits)
    max_value = 2 ** num_value_bits - 1
    return (min_value, max_value)


def uint_get_min_max(num_bits, frac_bits):
    min_value = 0
    max_value = (2 ** num_bits - 1) / (2 ** frac_bits)
    return (min_value, max_value)


def int_get_min_max(num_bits, frac_bits):
    # Need to subtract one bit for the sign bit
    num_value_bits = num_bits - 1
    min_value = -(2 ** num_value_bits) / (2 ** frac_bits)
    max_value = (2 ** num_value_bits - 1) / (2 ** frac_bits)
    return (min_value, max_value)


# Activation can be signed or unsiged
def fake_quant_op_activation(x, quantization_config, signed_output):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_activation"]
    if signed_output:
        min_val, max_val = int_get_min_max(num_bits, frac_bits)
    else:
        min_val, max_val = uint_get_min_max(num_bits, frac_bits)
    # print("Op fake quant activation traced, clip at [{:0.4f}, {:0.4f}]".format(min_val,max_val))
    return tf.quantization.fake_quant_with_min_max_args(
        x, min=min_val, max=max_val, num_bits=num_bits
    )


# Activation can be signed or unsiged
def fake_quant_bounds_activation(quantization_config, signed_output):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_activation"]
    if signed_output:
        min_val, max_val = int_get_min_max(num_bits, frac_bits)
    else:
        min_val, max_val = uint_get_min_max(num_bits, frac_bits)
    return min_val, max_val


# Weight is always signed
def fake_quant_op_weight(w, quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_weights"]
    min_val, max_val = int_get_min_max(num_bits, frac_bits)
    return tf.quantization.fake_quant_with_min_max_args(
        w, min=min_val, max=max_val, num_bits=num_bits
    )


# Bias is always signed
def fake_quant_op_bias(b, quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_bias"]
    min_val, max_val = int_get_min_max(num_bits, frac_bits)
    return tf.quantization.fake_quant_with_min_max_args(
        b, min=min_val, max=max_val, num_bits=num_bits
    )


# Input is always uint
def fake_quant_op_input(x, quantization_config):
    num_bits = quantization_config["input_bits"]
    frac_bits = num_bits - quantization_config["int_bits_input"]
    min_val, max_val = uint_get_min_max(num_bits, frac_bits)
    return tf.quantization.fake_quant_with_min_max_args(
        x, min=min_val, max=max_val, num_bits=num_bits
    )


def downscale_op_input(x, quantization_config):
    num_bits = quantization_config["input_bits"]
    frac_bits = num_bits - quantization_config["int_bits_input"]
    min_val, max_val = uint_get_min_max(num_bits, frac_bits)
    # print("ds input called {} -> {:0.4f}".format(2**num_bits-1,max_val))
    return x * max_val / (2 ** num_bits - 1)


if __name__ == "__main__":
    x = 0.515625
    h = 0.5078125
    print("x: ", str(quantize_uint(x, 8, 6)))
    print("h: ", str(quantize_uint(h, 8, 6)))

