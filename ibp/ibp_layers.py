import tensorflow as tf
import numpy as np
from quantized_layers import QuantizableLayer, QuantizedDense, QuantizedConv2D

assert tf.__version__.startswith("2."), "Please upgrade to TensorFlow 2.x"


class BoundPropDenseFusedWithSpec(tf.keras.layers.Layer):
    def __init__(self, output_layer_weights, weight_transform_matrix, **kwargs):

        self.output_layer_weights = output_layer_weights
        self.weight_transform_matrix = weight_transform_matrix
        super(BoundPropDenseFusedWithSpec, self).__init__(**kwargs)

    @property
    def units(self):
        return int(self.weight_transform_matrix.shape[1])

    def build(self, input_shape):
        self.trainable_weights.extend(self.output_layer_weights)
        super(BoundPropDenseFusedWithSpec, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs):
        lower_bound_head, upper_bound_head = inputs[0], inputs[1]

        W_orig = self.output_layer_weights[0]
        # Fuse matrices
        W = tf.matmul(W_orig, self.weight_transform_matrix)

        # Check if there exists a bias term
        if len(self.output_layer_weights) > 1:
            b_orig = self.output_layer_weights[1]
            # Fuse with bias term
            b = tf.linalg.matvec(self.weight_transform_matrix, b_orig, transpose_a=True)
        else:
            b = tf.constant(0, dtype=tf.float32)

        # Center and width
        center_prev = 0.5 * (upper_bound_head + lower_bound_head)
        edge_len_prev = 0.5 * (upper_bound_head - lower_bound_head)

        # Two matrix multiplications
        center = tf.matmul(center_prev, W) + b
        edge_len = tf.matmul(edge_len_prev, tf.abs(W))  # Edge length has no bias

        # New bounds
        lower_bound_head = center - edge_len
        upper_bound_head = center + edge_len

        outputs = [lower_bound_head, upper_bound_head]

        return outputs


class BoundPropMinMaxLowerBoundFusion(tf.keras.layers.Layer):
    def __init__(self, **kwargs):

        super(BoundPropMinMaxLowerBoundFusion, self).__init__(**kwargs)

    def call(self, inputs):
        max_list = []
        for lb, ub in inputs:
            # Fuse on lower bound
            m = tf.reduce_min(lb, axis=-1)
            max_list.append(m)

        min_tensors = tf.stack(max_list, axis=-1)
        return tf.reduce_max(min_tensors, axis=-1)


@tf.function
def only_negatives(x):
    s = tf.sign(x)
    return x * (1.0 - s) * 0.5


class BoundPropRelevantLowerBoundFusion(tf.keras.layers.Layer):
    def __init__(self, bound_margin, **kwargs):
        self.bound_margin = bound_margin
        super(BoundPropRelevantLowerBoundFusion, self).__init__(**kwargs)

    def call(self, inputs):
        max_list = []
        for lb, ub in inputs:
            # Fuse on lower bound
            if self.bound_margin > 0.0:
                lb = lb - self.bound_margin
            relevant_lbs = only_negatives(lb)
            m = tf.reduce_sum(relevant_lbs, axis=-1)

            max_list.append(m)

        min_tensors = tf.stack(max_list, axis=-1)

        return tf.reduce_sum(min_tensors, axis=-1)


class BoundPropGenericLayer(tf.keras.layers.Layer):
    def __init__(self, base_layer, **kwargs):

        self.base_layer = base_layer
        super(BoundPropGenericLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Simply pass through the original activation layer
        return [self.base_layer(inputs[0]), self.base_layer(inputs[1])]


class BoundPropQuantizedDense(tf.keras.layers.Layer):
    def __init__(self, ff_layer, **kwargs):

        self.ff_layer = ff_layer
        super(BoundPropQuantizedDense, self).__init__(**kwargs)

    @property
    def units(self):
        return self.ff_layer.units

    def build(self, input_shape):
        self.trainable_weights.extend(self.ff_layer.trainable_weights)
        super(BoundPropQuantizedDense, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        lower_bound_head, upper_bound_head = inputs[0], inputs[1]

        W = self.ff_layer.quantized_kernel
        b = self.ff_layer.quantized_bias

        # Center and width
        center_prev = 0.5 * (upper_bound_head + lower_bound_head)
        edge_len_prev = 0.5 * (upper_bound_head - lower_bound_head)

        # Two matrix multiplications
        center = tf.matmul(center_prev, W) + b
        edge_len = tf.matmul(edge_len_prev, tf.abs(W))  # Edge length has no bias

        # New bounds
        lower_bound_head = center - edge_len
        upper_bound_head = center + edge_len
        # tf.print("\nedge len:\n",edge_len)

        upper_bound_head = self.ff_layer.quantized_activation(upper_bound_head)
        lower_bound_head = self.ff_layer.quantized_activation(lower_bound_head)

        outputs = [lower_bound_head, upper_bound_head]

        return outputs


class BoundPropQuantizedConv2D(tf.keras.layers.Layer):
    def __init__(self, ff_layer, **kwargs):

        self.ff_layer = ff_layer
        super(BoundPropQuantizedConv2D, self).__init__(**kwargs)

    @property
    def strides(self):
        return self.ff_layer.strides

    @property
    def padding(self):
        return self.ff_layer.padding

    def build(self, input_shape):
        self.trainable_weights.extend(self.ff_layer.trainable_weights)
        super(BoundPropQuantizedConv2D, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        lower_bound_head, upper_bound_head = inputs[0], inputs[1]

        W = self.ff_layer.quantized_kernel
        b = self.ff_layer.quantized_bias

        # Center and width
        center_prev = 0.5 * (upper_bound_head + lower_bound_head)
        edge_len_prev = 0.5 * (upper_bound_head - lower_bound_head)

        # Two matrix multiplications
        center = (
            tf.nn.conv2d(
                center_prev, W, strides=self.strides, padding=self.padding.upper()
            )
            + b
        )
        # Edge length has no bias
        edge_len = tf.nn.conv2d(
            edge_len_prev, tf.abs(W), strides=self.strides, padding=self.padding.upper()
        )

        # New bounds
        lower_bound_head = center - edge_len
        upper_bound_head = center + edge_len

        upper_bound_head = self.ff_layer.quantized_activation(upper_bound_head)
        lower_bound_head = self.ff_layer.quantized_activation(lower_bound_head)

        outputs = [lower_bound_head, upper_bound_head]

        return outputs