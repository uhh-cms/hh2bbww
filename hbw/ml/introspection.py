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


def get_gradients(model: tf.keras.models.Model, inputs: np.array, output_node: int = 0) -> np.array:
    """
    Calculate gradients of *model* between batch-normalized *inputs* and pre-softmax output *output_node*

    :param model: The Keras model for which the gradients are to be calculated. The first layer of
    the model is expected to be a BatchNormalization layer.
    :type model: keras.Model
    :param inputs: The input data for the model. This should be a numpy array of the inputs to the
    model, which will be batch-normalized before being passed through the model.
    :type inputs: np.array
    :param output_node: The index of the output node for which the gradients are to be calculated.
    This refers to the index of the node in the final layer of the model, before the softmax activation. Defaults to 0.
    :type output_node: int, optional
    :return: A numpy array of the gradients of the model with respect to the inputs.
    The shape of this array will be the same as the shape of the inputs.
    :rtype: np.array
    :raises Exception: If the first layer of the model is not a BatchNormalization layer, an exception is raised.
    """
    batch_norm = model.layers[0]
    if batch_norm.__class__.__name__ != "BatchNormalization":
        raise Exception(f"First layer is expected to be BatchNormalization but is {batch_norm.__class__.__name__}")
    inp = batch_norm(tf.convert_to_tensor(inputs, dtype=tf.float32))
    with tf.GradientTape() as tape:
        tape.watch(inp)
        outp = inp
        for layer in model.layers[1:]:
            layer = copy.copy(layer)
            if layer.name == model.layers[-1].name:
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
    gradients = get_gradients(model, inputs, output_node)

    sum_gradients = np.sum(np.abs(gradients), axis=0)
    sum_gradients = dict(zip(input_features, sum_gradients))
    sum_gradients = dict(sorted(sum_gradients.items(), key=lambda x: abs(x[1]), reverse=True))

    return sum_gradients


def gradient_times_input(model, inputs, output_node: int = 0, input_features: list | None = None):
    """
    Gradient * Input of *model* between batch-normalized *inputs* and pre-softmax output *output_node*
    """
    gradients = get_gradients(model, inputs, output_node)

    # sum_gradients = np.sum(np.abs(gradients, axis=0))
    sum_gradients = np.abs(np.sum(gradients, axis=0))
    sum_gradients = dict(zip(input_features, sum_gradients))
    sum_gradients = dict(sorted(sum_gradients.items(), key=lambda x: abs(x[1]), reverse=True))

    return sum_gradients


def shap_ranking(model, inputs, output_node: int = 0, input_features: list | None = None):
    # create an explainer function using the background distribution 'X100'
    X100 = shap.utils.sample(inputs, 100)
    predict = functools.partial(model.predict, verbose=0)
    explainer = shap.Explainer(predict, X100)

    # calculate shap values
    shap_values = explainer(inputs[:20])
    shap_values.feature_names = list(input_features)
    shap_values.shape

    shap_ranking = dict(zip(shap_values.feature_names, shap_values[:, :, output_node].abs.mean(axis=0).values))
    shap_ranking = dict(sorted(shap_ranking.items(), key=lambda x: abs(x[1]), reverse=True))

    return shap_ranking, shap_values
