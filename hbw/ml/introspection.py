# coding: utf-8
"""
Functions for model introspection.
"""

from __future__ import annotations

import copy
import functools
import tensorflow as tf
import shap
import numpy as np


def get_input_post_batchnorm(model: tf.keras.models.Model, inputs: np.array) -> np.array:
    """
    Get the input data after passing through the batch normalization layer of *model*

    :param model: The Keras model for which the input data is to be retrieved. The first layer of
    the model is expected to be a BatchNormalization layer.
    :param inputs: The input data for the model. This should be a numpy array of the inputs to the
    model, which will be batch-normalized before being passed through the model.
    :return: A numpy array of the input data after passing through the batch normalization layer.
    The shape of this array will be the same as the shape of the inputs.
    :raises Exception: If the first layer of the model is not a BatchNormalization layer, an exception is raised.
    """
    batch_norm = model.layers[0]
    if batch_norm.__class__.__name__ != "BatchNormalization":
        raise Exception(f"First layer is expected to be BatchNormalization but is {batch_norm.__class__.__name__}")
    inp = batch_norm(tf.convert_to_tensor(inputs, dtype=tf.float32))
    return inp


def get_gradients(
    model: tf.keras.models.Model,
    inputs: np.array,
    output_node: int = 0,
    skip_batch_norm: bool = False,
) -> np.array:
    """
    Calculate gradients of *model* between batch-normalized *inputs* and pre-softmax output *output_node*

    :param model: The Keras model for which the gradients are to be calculated. The first layer of
    the model is expected to be a BatchNormalization layer.
    :param inputs: The input data for the model. This should be a numpy array of the inputs to the
    model, which will be batch-normalized before being passed through the model.
    :param output_node: The index of the output node for which the gradients are to be calculated.
    This refers to the index of the node in the final layer of the model, before the softmax activation. Defaults to 0.
    :param skip_batch_norm: If True, the input data is not passed through the batch normalization layer.
    :return: A numpy array of the gradients of the model with respect to the inputs.
    The shape of this array will be the same as the shape of the inputs.
    """
    if skip_batch_norm:
        inp = get_input_post_batchnorm(model, inputs)
        layers = model.layers[1:]
    else:
        inp = tf.convert_to_tensor(inputs, dtype=tf.float32)
        layers = model.layers
    with tf.GradientTape() as tape:
        tape.watch(inp)
        outp = inp
        for layer in layers:
            layer = copy.copy(layer)
            if layer.name == layers[-1].name:
                # for the final layer, copy the layer and remove the softmax activation
                layer = copy.copy(layer)
                layer.activation = None
            outp = layer(outp)
            out_i = outp[:, output_node]

    gradients = tape.gradient(out_i, inp)
    return gradients


def sensitivity_analysis(model, inputs, output_node: int = 0, input_features: list | None = None):
    """
    Sensitivity analysis of *model* between batch-normalized *inputs* and pre-softmax output *output_node*
    """
    gradients = get_gradients(model, inputs, output_node, skip_batch_norm=True)

    sum_gradients = np.sum(np.abs(gradients), axis=0)
    sum_gradients = dict(zip(input_features, sum_gradients))
    sum_gradients = dict(sorted(sum_gradients.items(), key=lambda x: abs(x[1]), reverse=True))

    return sum_gradients


def gradient_times_input(model, inputs, output_node: int = 0, input_features: list | None = None):
    """
    Gradient * Input of *model* between batch-normalized *inputs* and pre-softmax output *output_node*
    """
    gradients = get_gradients(model, inputs, output_node, skip_batch_norm=True)
    inputs = get_input_post_batchnorm(model, inputs)

    # NOTE: remove np.abs?
    sum_grad_times_inp = np.abs(np.sum(gradients * inputs, axis=0))
    sum_grad_times_inp = dict(zip(input_features, sum_grad_times_inp))
    sum_grad_times_inp = dict(sorted(sum_grad_times_inp.items(), key=lambda x: abs(x[1]), reverse=True))

    return sum_grad_times_inp


def shap_ranking(model, inputs, output_node: int = 0, input_features: list | None = None):
    # create an explainer function using the background distribution 'X100'
    X100 = shap.utils.sample(inputs, 100)
    predict = functools.partial(model.predict, verbose=0)
    explainer = shap.Explainer(predict, X100)

    # calculate shap values
    shap_values = explainer(inputs[:100])
    shap_values.feature_names = list(input_features)

    shap_ranking = dict(zip(shap_values.feature_names, shap_values[:, :, output_node].abs.mean(axis=0).values))
    shap_ranking = dict(sorted(shap_ranking.items(), key=lambda x: abs(x[1]), reverse=True))

    return shap_ranking, shap_values
