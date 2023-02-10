import tensorflow as tf
import numpy as np

assert tf.__version__.startswith("2."), "Please upgrade to TensorFlow 2.x"


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, residual_layers, **kwargs):
        self.residual_layers = residual_layers
        super().__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        for l in self.residual_layers:
            # Variables created here will be added to the
            # l.build(x.shape)
            # self.trainable_variables.append(l.trainable_variables)
            x = l(x)

        # @Hack: to compare if two shapes are equal (__eq__ operator does not work)
        if not x.shape.as_list() == inputs.shape.as_list():
            raise ValueError(
                "Error! Input and last layers of a resesidual blocks must have equal shape. Got ({}) and ({})".format(
                    inputs.shape, x.shape
                )
            )
        y = x + inputs
        return y


@tf.function
def get_single_batchnorm_transform(mean, variance, gamma, beta):
    C = gamma / variance
    D = beta - mean / variance
    return C, D


# TODO: Create one generic layer that just applies the op to both bounds
class BoundPropFlatten(tf.keras.layers.Layer):
    def __init__(self, flatten_layer, **kwargs):

        self.flatten_layer = flatten_layer
        super(BoundPropFlatten, self).__init__(**kwargs)

    def call(self, inputs):
        # Simply pass through the original flatten layer
        outputs = []
        for i in inputs:
            o = self.flatten_layer(i)
            outputs.append(o)
        return outputs


class BoundPropReshape(tf.keras.layers.Layer):
    def __init__(self, reshape_layer, **kwargs):

        self.reshape_layer = reshape_layer
        super(BoundPropReshape, self).__init__(**kwargs)

    def call(self, inputs):
        # Simply pass through the original reshape layer
        outputs = []
        for i in inputs:
            o = self.reshape_layer(i)
            outputs.append(o)
        return outputs


class BoundPropActivation(tf.keras.layers.Layer):
    def __init__(self, activation_layer, **kwargs):

        self.activation_layer = activation_layer
        super(BoundPropActivation, self).__init__(**kwargs)

    def call(self, inputs):
        # Simply pass through the original activation layer
        return [self.activation_layer(inputs[0]), self.activation_layer(inputs[1])]


class BoundPropGenericLayer(tf.keras.layers.Layer):
    def __init__(self, base_layer, **kwargs):

        self.base_layer = base_layer
        super(BoundPropGenericLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Simply pass through the original activation layer
        return [self.base_layer(inputs[0]), self.base_layer(inputs[1])]


class BoundPropGlobalMaxPool2D(tf.keras.layers.Layer):
    def __init__(self, global_max_pool2d_layer, **kwargs):

        self.global_max_pool2d_layer = global_max_pool2d_layer
        super(BoundPropGlobalMaxPool2D, self).__init__(**kwargs)

    def call(self, inputs):
        # Simply pass through the original layer
        return [
            self.global_max_pool2d_layer(inputs[0]),
            self.global_max_pool2d_layer(inputs[1]),
        ]


class BoundPropMaxPool2D(tf.keras.layers.Layer):
    def __init__(self, max_pool_layer, **kwargs):

        self.max_pool_layer = max_pool_layer
        super(BoundPropMaxPool2D, self).__init__(**kwargs)

    def call(self, inputs):
        # Simply pass through the original layer
        return [self.max_pool_layer(inputs[0]), self.max_pool_layer(inputs[1])]


class BoundPropBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, batchnorm_layer, **kwargs):

        self.batchnorm_layer = batchnorm_layer
        super(BoundPropBatchNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        # Simply pass through the original activation layer
        self.batchnorm_layer.epsilon
        self.batchnorm_layer.center
        self.batchnorm_layer.gamma
        self.batchnorm_layer.beta
        self.batchnorm_layer.moving_variance
        self.batchnorm_layer.moving_mean

        # Bachnorm layer implemented by DeepMind:
        # https://github.com/deepmind/interval-bound-propagation/blob/master/interval_bound_propagation/src/bounds.py
        multiplier = tf.math.rsqrt(
            self.batchnorm_layer.moving_variance + self.batchnorm_layer.epsilon
        )
        if self.batchnorm_layer.scale is not None:
            multiplier *= self.batchnorm_layer.gamma
        w = multiplier
        # Element-wise bias.
        b = -multiplier * self.batchnorm_layer.moving_mean
        if self.batchnorm_layer.beta is not None:
            b += self.batchnorm_layer.beta
        # print("b: ",str(b.shape))
        # print("w: ",str(w.shape))
        # print("multiplier: ",str(multiplier.shape))
        # b = tf.squeeze(b, axis=0)
        # Because the scale might be negative, we need to apply a strategy similar
        # to linear.
        c = (inputs[0] + inputs[1]) / 2.0
        r = (inputs[1] - inputs[0]) / 2.0
        c = tf.multiply(c, w) + b
        r = tf.multiply(r, tf.abs(w))
        return [c - r, c + r]


class BoundPropResidualBlock(tf.keras.layers.Layer):
    def __init__(self, residual_block, **kwargs):

        self.residual_block = residual_block

        super(BoundPropResidualBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights.extend(self.residual_block.trainable_weights)
        super(BoundPropResidualBlock, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs):

        head = inputs
        for l in self.residual_block.residual_layers:
            bound_layer = build_bound_layer(l)
            head = bound_layer(head)

        outputs = [head[0] + inputs[0], head[1] + inputs[1]]
        return outputs


class BoundPropConv1D(tf.keras.layers.Layer):
    def __init__(self, conv1d_layer, **kwargs):

        self.conv1d_layer = conv1d_layer

        super(BoundPropConv1D, self).__init__(**kwargs)

    @property
    def strides(self):
        return self.conv1d_layer.strides

    @property
    def padding(self):
        return self.conv1d_layer.padding

    def build(self, input_shape):
        self.trainable_weights.extend(self.conv1d_layer.trainable_weights)
        super(BoundPropConv1D, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs):
        lower_bound_head, upper_bound_head = inputs[0], inputs[1]

        W, b = self.conv1d_layer.weights

        # Center and width
        center_prev = 0.5 * (upper_bound_head + lower_bound_head)
        edge_len_prev = 0.5 * (upper_bound_head - lower_bound_head)

        # Two matrix multiplications
        center = (
            tf.nn.conv1d(
                center_prev, W, stride=self.strides, padding=self.padding.upper()
            )
            + b
        )
        # Edge length has no bias
        edge_len = tf.nn.conv1d(
            edge_len_prev, tf.abs(W), stride=self.strides, padding=self.padding.upper()
        )

        # New bounds
        lower_bound_head = center - edge_len
        upper_bound_head = center + edge_len

        upper_bound_head = self.conv1d_layer.activation(upper_bound_head)
        lower_bound_head = self.conv1d_layer.activation(lower_bound_head)

        outputs = [lower_bound_head, upper_bound_head]

        return outputs


class BoundPropConv2D(tf.keras.layers.Layer):
    def __init__(self, conv2d_layer, fused_batchnorm_layer=None, **kwargs):

        self.conv2d_layer = conv2d_layer
        self.fused_batchnorm_layer = fused_batchnorm_layer
        if (
            not self.fused_batchnorm_layer is None
        ) and self.conv2d_layer.activation != tf.keras.activations.linear:
            raise ValueError(
                "Cannot fuse conv2d layer with batch_norm layer if the conv2d_layer has a non-linear activation function"
            )
        super(BoundPropConv2D, self).__init__(**kwargs)

    @property
    def strides(self):
        return self.conv2d_layer.strides

    @property
    def padding(self):
        return self.conv2d_layer.padding

    def build(self, input_shape):
        self.trainable_weights.extend(self.conv2d_layer.trainable_weights)
        if not self.fused_batchnorm_layer is None:
            self.trainable_weights.extend(self.fused_batchnorm_layer.trainable_weights)

        super(BoundPropConv2D, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs):
        lower_bound_head, upper_bound_head = inputs[0], inputs[1]

        W = self.conv2d_layer.weights[0]
        # bias might be disabled in some convolutional architecutres
        if self.conv2d_layer.use_bias:
            b = self.conv2d_layer.weights[1]
        else:
            b = tf.constant(0, dtype=tf.float32)

        if not self.fused_batchnorm_layer is None:
            # Fuse conv2d with batchnorm weights to keep intervals tight
            C, d = get_single_batchnorm_transform(
                variance=self.fused_batchnorm_layer.moving_variance,
                mean=self.fused_batchnorm_layer.moving_mean,
                gamma=self.fused_batchnorm_layer.gamma,
                beta=self.fused_batchnorm_layer.beta,
            )
            new_W = W * C
            new_b = b * C + d
            W = new_W
            b = new_b

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

        upper_bound_head = self.conv2d_layer.activation(upper_bound_head)
        lower_bound_head = self.conv2d_layer.activation(lower_bound_head)

        outputs = [lower_bound_head, upper_bound_head]

        return outputs


class BoundPropDense(tf.keras.layers.Layer):
    def __init__(self, dense_layer, disable_activation=None, **kwargs):

        self.dense_layer = dense_layer
        self.disable_activation = disable_activation
        super(BoundPropDense, self).__init__(**kwargs)

    @property
    def units(self):
        return self.dense_layer.units

    def build(self, input_shape):
        self.trainable_weights.extend(self.dense_layer.trainable_weights)
        super(BoundPropDense, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs):
        lower_bound_head, upper_bound_head = inputs[0], inputs[1]

        W = self.dense_layer.weights[0]
        if self.dense_layer.use_bias:
            b = self.dense_layer.weights[1]
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
        # tf.print("\nedge len:\n",edge_len)

        if not self.disable_activation:
            upper_bound_head = self.dense_layer.activation(upper_bound_head)
            lower_bound_head = self.dense_layer.activation(lower_bound_head)

        outputs = [lower_bound_head, upper_bound_head]

        return outputs


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

