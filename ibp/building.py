import numpy as np
import tensorflow as tf
import ibp.ibp_layers as ibp_layers
import ibp.utils as ibp_utils

""" Checks if the given keras model is supported by ABD-SMT,
    raises an error if the model contains a not supported layer
"""


# allowed_layer_types = [
#     tf.keras.layers.Dense,
#     tf.keras.layers.Lambda,
#     tf.keras.layers.Activation,
#     tf.keras.layers.BatchNormalization,
#     tf.keras.layers.GlobalMaxPooling2D,
#     tf.keras.layers.Conv1D,
#     tf.keras.layers.Conv2D,
#     tf.keras.layers.MaxPool2D,
#     tf.keras.layers.MaxPooling2D,
#     tf.keras.layers.Convolution1D,
#     tf.keras.layers.Convolution2D,
#     tf.keras.layers.Flatten,
#     tf.keras.layers.Reshape,
#     tf.keras.layers.Dropout,
#     tf.keras.layers.InputLayer,
#     ibp_layers.ResidualBlock,
#     ibp_quanitzed.QuantizedDense,
#     ibp_quanitzed.InputQuantizationLayer,
#     ibp_quanitzed.QuantizedConv2D,
# ]
#

# def build_bound_layer(layer):
#     if type(layer) is tf.keras.layers.Dense:
#         return ibp_layers.BoundPropDense(layer)
#     elif type(layer) is tf.keras.layers.Flatten:
#         return ibp_layers.BoundPropFlatten(layer)
#     elif type(layer) is tf.keras.layers.Reshape:
#         return ibp_layers.BoundPropReshape(layer)
#     elif type(layer) is tf.keras.layers.Activation:
#         return ibp_layers.BoundPropActivation(layer)
#     elif type(layer) is tf.keras.layers.Lambda:
#         return ibp_layers.BoundPropGenericLayer(layer)
#     elif type(layer) is ibp_layers.ResidualBlock:
#         return ibp_layers.BoundPropResidualBlock(layer)
#     elif type(layer) is tf.keras.layers.BatchNormalization:
#         return ibp_layers.BoundPropBatchNormalization(layer)
#     elif (
#         type(layer) is tf.keras.layers.GlobalMaxPool2D
#         or type(layer) is tf.keras.layers.GlobalMaxPooling2D
#     ):
#         return ibp_layers.BoundPropGlobalMaxPool2D(layer)
#     elif (
#         type(layer) is tf.keras.layers.MaxPool2D
#         or type(layer) is tf.keras.layers.MaxPooling2D
#     ):
#         return ibp_layers.BoundPropMaxPool2D(layer)
#     elif (
#         type(layer) is tf.keras.layers.Conv1D
#         or type(layer) is tf.keras.layers.Convolution1D
#     ):
#         return ibp_layers.BoundPropConv1D(layer)
#     elif (
#         type(layer) is tf.keras.layers.Conv2D
#         or type(layer) is tf.keras.layers.Convolution2D
#     ):
#         return ibp_layers.BoundPropConv2D(layer)
#     elif type(layer) is ibp_quanitzed.QuantizedDense:
#         return ibp_quanitzed.BoundPropQuantizedDense(layer)
#     elif type(layer) is ibp_quanitzed.QuantizedConv2D:
#         return ibp_quanitzed.BoundPropQuantizedConv2D(layer)
#     elif type(layer) is ibp_quanitzed.InputQuantizationLayer:
#         return ibp_quanitzed.BoundPropInputQuantization(layer)
#     return None


def build_bound_model(ff_qnn):

    # Remove batch dim
    input_lower_bound = tf.keras.layers.Input(shape=ff_qnn._input_shape[1:])
    input_upper_bound = tf.keras.layers.Input(shape=ff_qnn._input_shape[1:])

    input_list = [input_lower_bound, input_upper_bound]

    head = [ff_qnn.preprocess(input_lower_bound), ff_qnn.preprocess(input_upper_bound)]

    if ff_qnn.input_reshape is not None:
        head = ibp_layers.BoundPropGenericLayer(ff_qnn.input_reshape)(head)
    regularization_loss = tf.constant(0.0, dtype=tf.float32)
    for c in ff_qnn.conv2d_layers:
        head = ibp_layers.BoundPropQuantizedConv2D(c)(head)
        regularization_loss += tf.reduce_mean(head[1] - head[0])

    head = ibp_layers.BoundPropGenericLayer(ff_qnn.flatten_layer)(head)

    for i in range(len(ff_qnn.dense_layers) - 1):
        head = ibp_layers.BoundPropQuantizedDense(ff_qnn.dense_layers[i])(head)
        regularization_loss += tf.reduce_mean(head[1] - head[0])

    output_layer = ff_qnn.dense_layers[-1]
    bound_model = tf.keras.Model(inputs=input_list, outputs=head)
    return bound_model, output_layer, regularization_loss


def add_single_spec_to_bound_model(
    pre_logit_bound_model, output_layer, allowed_classes, reduce_max, bound_margin
):
    """
    Fuses the output layer with the specification and adds it to the
     the bound model.
    """
    # Maps output layer to 1 variable (spec violation bound)
    # size of the last hidden layer
    pre_logit_size = int(pre_logit_bound_model.outputs[0].shape[-1])
    output_size = int(output_layer.units)
    forbidden_classes = [i for i in range(output_size) if not i in allowed_classes]

    # For each allowed class we create a new spec matrix that we will merge
    #  with the output layer to obtain tighter bounds
    # print("add_single_spec_to_bound_model")
    # print("allowed          : ",str(allowed_classes))
    # print("forbidden_classes: ",str(forbidden_classes))
    # print("pre_logit_bound_model: ",str(pre_logit_bound_model))
    # print("output_layer: ",str(output_layer))
    all_bounds = []
    for yes in allowed_classes:
        W_mask = np.zeros([output_size, len(forbidden_classes)], dtype=np.float32)
        for i, no in enumerate(forbidden_classes):
            # We need index of current position + position of forbidden class
            W_mask[yes, i] = 1
            W_mask[no, i] = -1
        W_mask = tf.constant(W_mask, dtype=tf.float32)
        head = ibp_layers.BoundPropDenseFusedWithSpec(
            [output_layer.quantized_kernel, output_layer.quantized_bias], W_mask
        )(pre_logit_bound_model.outputs)
        all_bounds.append(head)

    # all_bounds is a list of vectors, exactly one for each allowed class
    if reduce_max:
        spec_fusion = ibp_layers.BoundPropMinMaxLowerBoundFusion()(all_bounds)
    else:
        spec_fusion = ibp_layers.BoundPropRelevantLowerBoundFusion(bound_margin)(
            all_bounds
        )

    return spec_fusion


def add_spec_list_to_bound_model(
    pre_logit_bound_model, output_layer, spec_list, reduce_max, bound_margin
):
    # Only one specification -> encapsulate spec into a list
    if not type(spec_list[0]) is list:
        spec_list = [spec_list]

    spec_bound_list = []
    for spec in spec_list:
        f = add_single_spec_to_bound_model(
            pre_logit_bound_model, output_layer, spec, reduce_max, bound_margin
        )
        spec_bound_list.append(f)

    if len(spec_bound_list) == 1:
        # Only one specification -> no need to merge (stack)
        stacked_output = spec_bound_list[0]
    else:
        stacked_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))(
            spec_bound_list
        )
    bound_model_with_spec = tf.keras.Model(
        inputs=pre_logit_bound_model.inputs, outputs=stacked_output
    )
    return bound_model_with_spec


# def add_robustness_spec_to_qnn_bound_model(
#     pre_logit_bound_model, output_layer, reduce_max, bound_margin
# ):
#     spec_bound_list = []
#     for i in range(output_layer.units):
#         f = add_single_spec_to_bound_model(
#             pre_logit_bound_model, output_layer, [i], reduce_max, bound_margin
#         )
#         spec_bound_list.append(f)
#
#     if len(spec_bound_list) == 1:
#         # Only one specification -> no need to merge (stack)
#         stacked_output = spec_bound_list[0]
#     else:
#         stacked_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))(
#             spec_bound_list
#         )
#     bound_model_with_spec = tf.keras.Model(
#         inputs=pre_logit_bound_model.inputs, outputs=stacked_output
#     )
#     return bound_model_with_spec
#

#
# def build_pure_bound_model(pre_logit_bound_model, output_layer):
#     new_outputs = ibp_layers.BoundPropDense(output_layer, disable_activation=True)(
#         pre_logit_bound_model.outputs
#     )
#     return tf.keras.Model(inputs=pre_logit_bound_model.inputs, outputs=new_outputs)