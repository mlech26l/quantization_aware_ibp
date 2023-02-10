import tensorflow as tf
from tensorflow.keras.layers import Layer
import quantization_util as qu
import numpy as np

if tf.__version__.startswith("1.") or tf.__version__.startswith("0."):
    raise ValueError("Please upgrade to TensorFlow 2.x")


class QuantizedModel(tf.keras.Model):
    def __init__(
        self,
        layers,
        input_bits,
        quantization_bits,
        dropout_rate=0.0,
        last_layer_signed=False,
        last_layer_no_act=False,
        input_reshape=None,
        dont_quantize_first=False,
        dont_quantize_last=False,
        **kwargs,
    ):

        self.quantization_config = {
            "input_bits": input_bits,
            "quantization_bits": quantization_bits,
            "int_bits_weights": 1,  # Signed, +-0.3
            "int_bits_bias": 2,  # Signed, +-1.2
            "int_bits_activation": 2,  # Unsigned (relu), +1.3
            "int_bits_input": 0,  # Unsigned (relu), +1.3
        }
        # Override
        for k, v in kwargs.items():
            if k in self.quantization_config.keys():
                self.quantization_config[k] = v
        # self.quantization_config = {
        #     "input_bits": input_bits,
        #     "quantization_bits": quantization_bits,
        #     "int_bits_weights": 1, # Signed, +-0.3
        #     "int_bits_bias": 2, # Signed, +-1.2
        #     "int_bits_activation": 1, # Unsigned (relu), +1.3
        #     "int_bits_input": 1, # Unsigned (relu), +1.3
        # }
        self._last_layer_signed = last_layer_signed
        self._last_layer_no_act = last_layer_no_act
        self.input_reshape = None
        self.input_reshape_shape = None

        super(QuantizedModel, self).__init__()

        if not input_bits in [4, 8]:
            raise ValueError("Only 4 and 8 bit inputs supported")
        self.conv2d_layers = []
        self.output_dim = layers[-1]
        self.dense_layers = []
        current_bits = input_bits
        for i, l in enumerate(layers):
            if type(l) == list or type(l) == tuple:
                if len(l) != 3:
                    raise ValueError("Expected 3-tuple, got {}-tuple".format(len(l)))

                self.conv2d_layers.append(
                    QuantizedConv2D(
                        filters=l[0],
                        input_bits=current_bits,
                        quantization_config=self.quantization_config,
                        kernel_size=l[1],
                        strides=(l[2], l[2]),
                    )
                )
            elif type(l) == int:
                signed = (
                    True if self._last_layer_signed and i == len(layers) - 1 else False
                )
                no_activation = (
                    True if self._last_layer_no_act and i == len(layers) - 1 else False
                )

                self.dense_layers.append(
                    QuantizedDense(
                        output_dim=l,
                        input_bits=current_bits,
                        quantization_config=self.quantization_config,
                        # stochastic_kernel_scale=stochastic_kernel_scale,
                        signed_output=signed,
                        no_activation=no_activation,
                    )
                )
            else:
                raise ValueError("Unexpected type {} ({})".format(type(l), str(l)))
            current_bits = quantization_bits
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.0:
            self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.flatten_layer = tf.keras.layers.Flatten()
        if input_reshape is not None:
            self.input_reshape_shape = input_reshape
            self.input_reshape = tf.keras.layers.Reshape(input_reshape)

    def quantized_activation(self, x):
        y = qu.fake_quant_op_activation(x, self.quantization_config, self.signed_output)
        return y

    def copy_from_non_quantized(self, model):
        self.set_weights(model.get_weights())

    def get_non_quantized_copy(self):
        layers = []

        layers.append(
            tf.keras.layers.Lambda(
                lambda x: qu.fake_quant_op_input(
                    qu.downscale_op_input(x, self.quantization_config),
                    self.quantization_config,
                )
            )
        )
        activation = lambda x: qu.fake_quant_op_activation(
            x, self.quantization_config, False
        )
        last_activation = lambda x: qu.fake_quant_op_activation(
            x, self.quantization_config, self._last_layer_signed
        )
        # activation = "relu"
        # last_activation = None
        if self.input_reshape_shape is not None:
            layers.append(tf.keras.layers.Reshape(self.input_reshape_shape))
        for l in self.conv2d_layers:
            layers.append(
                tf.keras.layers.Conv2D(
                    l.filters, l.kernel_size, l.strides, activation=activation
                )
            )
        layers.append(tf.keras.layers.Flatten())
        for l in self.dense_layers[:-1]:
            layers.append(tf.keras.layers.Dense(l.units, activation=activation))
            layers.append(tf.keras.layers.Dropout(self.dropout_rate))
        layers.append(
            tf.keras.layers.Dense(
                self.dense_layers[-1].units, activation=last_activation
            )
        )
        return tf.keras.Sequential(layers)

    def build(self, input_shape):
        # print("Build called with input_shape: ",str(input_shape))
        self._input_shape = input_shape
        super(QuantizedModel, self).build(
            input_shape
        )  # Be sure to call this at the end

    def preprocess(self, x):
        x = qu.downscale_op_input(x, self.quantization_config)
        x = qu.fake_quant_op_input(x, self.quantization_config)
        return x

    def call(self, inputs, **kwargs):
        x = self.preprocess(inputs)
        if self.input_reshape is not None:
            x = self.input_reshape(x)
        for c in self.conv2d_layers:
            x = c(x)

        x = self.flatten_layer(x)

        for c in self.dense_layers:
            if self.dropout_rate > 0.0:
                x = self.dropout_layer(x)
            x = c(x)

        return x

    def call_and_collect(self, inputs):
        x = self.preprocess(inputs)
        hiddens = []
        for c in self.conv2d_layers:
            (x, h) = c.call_and_collect(x)
            hiddens.append(h)

        x = self.flatten_layer(x)

        for c in self.dense_layers:
            if self.dropout_rate > 0.0:
                x = self.dropout_layer(x)
            x, h = c.call_and_collect(x)
            hiddens.append(h)

        return x, hiddens

    def as_pure_fully_connected(self):
        dense_layers = []
        dense_weights = []
        current_size = (
            self.input_reshape_shape[0]
            * self.input_reshape_shape[1]
            * self.input_reshape_shape[2]
        )
        input_dim = current_size
        for c in self.conv2d_layers:
            stride = c.strides[0]
            w_conv, b_conv = c.kernel.numpy(), c.bias.numpy()
            w, b, current_size = conv2fc(w_conv, b_conv, current_size, stride)
            dense_layers.append(current_size)
            dense_weights.append((w, b))
        for c in self.dense_layers:
            dense_layers.append(c.units)
            dense_weights.append((c.kernel, c.bias))
        print("dense_layers: ", dense_layers)
        pure_dense = QuantizedModel(
            dense_layers,
            self.quantization_config["input_bits"],
            self.quantization_config["quantization_bits"],
            last_layer_signed=self._last_layer_signed,
        )
        # Instantiate model
        pure_dense(tf.zeros((1, input_dim)))

        # Overwrite weights
        for i in range(len(pure_dense.dense_layers)):
            w, b = dense_weights[i]
            pure_dense.dense_layers[i].kernel.assign(w)
            pure_dense.dense_layers[i].bias.assign(b)
        return pure_dense


def conv2fc(w_conv, b_conv, input_size, stride):
    # print("wconv_ shape: ",str(w_conv.shape))
    num_channels = int(w_conv.shape[2])
    input_map_dim = int(np.sqrt(input_size // num_channels))
    # print("Is dimension [{},{},{}] correct?".format(input_map_dim,input_map_dim,num_channels))
    assert (
        input_map_dim * input_map_dim * num_channels == input_size
    ), "Something is wrong here"
    num_filters = int(w_conv.shape[3])

    kernel_dim = int(w_conv.shape[0])
    # print("Num filters {}, correct?",str(num_filters))
    # print("Kernel dim {}, correct?",str(kernel_dim))

    # Stride = 2
    shifts = (input_map_dim - kernel_dim) // stride + 1
    # print("Num of shifts: {} correct?".format(shifts))

    w_index = np.arange(input_size)
    w_index = w_index.reshape([input_map_dim, input_map_dim, num_channels])

    num_units = shifts * shifts * num_filters
    # print("Total neurons of next layer: {}, correct?".format(num_units))
    w = np.zeros([input_size, num_units], dtype=w_conv.dtype)
    b = np.zeros([num_units], dtype=b_conv.dtype)
    for shift_1 in range(shifts):
        for shift_2 in range(shifts):
            for filter_i in range(num_filters):
                target_index = filter_i + num_filters * (shift_2 + shifts * shift_1)

                start_x = shift_1 * stride
                start_y = shift_2 * stride

                b[target_index] = b_conv[filter_i]
                for x in range(kernel_dim):
                    for y in range(kernel_dim):
                        for ch in range(num_channels):
                            # print("read W[{},{},{},{}]".format(x,y,ch,filter_i))
                            # print("Neuron ({}) += w*x[{},{},{}]".format(target_index,start_x+x,start_y+y,ch))
                            weight_value = w_conv[x, y, ch, filter_i]
                            w[
                                w_index[start_x + x, start_y + y, ch],
                                target_index,
                            ] = weight_value
    return w, b, num_units


class QuantizableLayer(Layer):
    def __init__(self, input_bits=None, quantization_config=None, **kwargs):
        if not input_bits in [4, 5, 6, 7, 8]:
            raise ValueError(
                "Input bit resolution '{}' not supported. (Supported: 4-8)".format(
                    input_bits
                )
            )
        self.input_bits = input_bits
        self.quantization_config = quantization_config

        super(QuantizableLayer, self).__init__(**kwargs)


class QuantizedDense(QuantizableLayer):
    def __init__(
        self,
        output_dim,
        signed_output=False,
        no_activation=False,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        **kwargs,
    ):

        self.units = output_dim
        self.kernel_initializer = kernel_initializer

        # We want to represent the output layer with a signed integer
        self.signed_output = signed_output
        self.no_activation = no_activation

        super(QuantizedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        print(f"Build Quantized Dense with {input_shape}")
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_shape[1]), self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.units],
            initializer=tf.keras.initializers.Constant(0.25),
            trainable=True,
        )

        super(QuantizedDense, self).build(
            input_shape
        )  # Be sure to call this at the end

    @property
    def quantized_kernel(self):
        return qu.fake_quant_op_weight(self.kernel, self.quantization_config)

    @property
    def quantized_bias(self):
        return qu.fake_quant_op_bias(self.bias, self.quantization_config)

    def call(self, x, training=None, **kwargs):
        # if(training is None):
        #     training = tf.keras.backend.learning_phase()

        # Fake quantize weights

        y = tf.matmul(x, self.quantized_kernel) + self.quantized_bias

        if not self.no_activation:
            y = self.quantized_activation(y)
        return y

    def quantized_activation(self, x):
        y = qu.fake_quant_op_activation(x, self.quantization_config, self.signed_output)
        return y

    def call_and_collect(self, x, training=None):
        # if(training is None):
        #     training = tf.keras.backend.learning_phase()

        # Fake quantize weights
        kernel = qu.fake_quant_op_weight(self.kernel, self.quantization_config)
        bias = qu.fake_quant_op_bias(self.bias, self.quantization_config)

        v = tf.matmul(x, kernel) + bias

        y = self.quantized_activation(v)
        return y, v

    def call_debug(self, x, training=None):
        # if(training is None):
        #     training = tf.keras.backend.learning_phase()

        # Fake quantize weights
        kernel = qu.fake_quant_op_weight(self.kernel, self.quantization_config)
        bias = qu.fake_quant_op_bias(self.bias, self.quantization_config)

        y = tf.matmul(x, kernel) + bias

        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_fixed_point_weights(self):
        kernel = qu.fake_quant_op_weight(self.kernel, self.quantization_config).numpy()
        bias = qu.fake_quant_op_bias(self.bias, self.quantization_config).numpy()
        return (kernel, bias)

    def get_quantized_weights(self):
        w, b = self.get_weights()
        return (
            qu.quantize_weight(w, self.quantization_config),
            qu.quantize_bias(b, self.quantization_config),
        )


class QuantizedConv2D(QuantizableLayer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        **kwargs,
    ):

        self.filters = filters

        # If dimension is integer assume symmetry
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)

        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = "VALID"
        super(QuantizedConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name="kernel",
            shape=[
                self.kernel_size[0],
                self.kernel_size[1],
                int(input_shape[3]),
                self.filters,
            ],
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.filters],
            initializer=tf.keras.initializers.Constant(0.25),
            trainable=True,
        )
        self._input_shape = input_shape
        self._output_shape = self.compute_output_shape(input_shape)
        super(QuantizedConv2D, self).build(
            input_shape
        )  # Be sure to call this at the end

    @property
    def quantized_kernel(self):
        return qu.fake_quant_op_weight(self.kernel, self.quantization_config)

    @property
    def quantized_bias(self):
        return qu.fake_quant_op_bias(self.bias, self.quantization_config)

    def call(self, x, **kwargs):
        # Fake quantize weights
        y = (
            tf.nn.conv2d(
                x,
                self.quantized_kernel,
                strides=[1, self.strides[0], self.strides[1], 1],
                padding=self.padding,
            )
            + self.quantized_bias
        )
        y = self.quantized_activation(y)
        return y

    def quantized_activation(self, x):
        y = qu.fake_quant_op_activation(x, self.quantization_config, False)
        return y

    def compute_output_shape(self, input_shape):
        # (W-K+2P)/S+1
        output_dim_1 = (input_shape[1] - self.kernel_size[0]) // self.strides[0] + 1
        output_dim_2 = (input_shape[2] - self.kernel_size[1]) // self.strides[1] + 1
        return (input_shape[0], output_dim_1, output_dim_2, self.filters)

    def get_quantized_weights(self):
        w, b = self.get_weights()
        return (
            qu.quantize_weight(w, self.quantization_config),
            qu.quantize_bias(b, self.quantization_config),
        )


if __name__ == "__main__":
    model = QuantizedModel([32, 16, 10], input_bits=4, quantization_bits=8)
    x_train = np.random.normal(size=(200, 5))
    y_train = np.random.normal(size=(200, 10))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    model.fit(x=x_train, y=y_train, batch_size=50, epochs=2)

    print("model.input_spec ", str(model.input_spec))
    print("model.input_shape", str(model.input_shape))