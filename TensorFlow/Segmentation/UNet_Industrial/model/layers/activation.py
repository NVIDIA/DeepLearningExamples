#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import tensorflow as tf

from model.layers.utils import _log_hparams

__all__ = ['crelu', 'elu', 'leaky_relu', 'prelu', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax', 'tanh']


def crelu(features, name='crelu', axis=-1):

    net = tf.nn.crelu(features, name=name, axis=axis)

    _log_hparams(classname='CReLU', layername=net.name, axis=axis, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net


def elu(features, name='elu'):

    net = tf.nn.elu(features, name=name)

    _log_hparams(classname='ELU', layername=net.name, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net


def leaky_relu(features, alpha=0.2, name='leaky_relu'):

    net = tf.nn.leaky_relu(features, alpha=alpha, name=name)

    _log_hparams(
        classname='LeakyReLU', layername=net.name, alpha=alpha, out_shape=str(net.get_shape()), out_dtype=net.dtype
    )

    return net


def prelu(inputs, channel_shared=False, trainable=True, name='prelu'):

    def parametric_relu(_x):

        if channel_shared:
            w_shape = (1, )

        else:
            w_shape = int(_x.get_shape()[-1])

        alphas = tf.get_variable(
            'alpha', w_shape, trainable=trainable, initializer=tf.initializers.truncated_normal(mean=-1.0, stddev=0.2)
        )

        alphas = tf.nn.sigmoid(alphas, name="constraining_alpha_var_in_0_1")

        return tf.maximum(_x, _x * alphas)

    with tf.variable_scope(name):
        net = parametric_relu(inputs)

    _log_hparams(
        classname='PReLU',
        layername=net.name,
        channel_shared=channel_shared,
        trainable=trainable,
        out_shape=str(net.get_shape()),
        out_dtype=net.dtype
    )

    return net


def relu(inputs, name='relu'):

    net = tf.nn.relu(inputs, name=name)

    _log_hparams(classname='ReLU', layername=net.name, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net


def relu6(inputs, name='relu6'):

    net = tf.nn.relu6(inputs, name=name)

    _log_hparams(classname='ReLU6', layername=net.name, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net


def selu(features, name='selu'):

    net = tf.nn.selu(features, name=name)

    _log_hparams(classname='SELU', layername=net.name, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net


def sigmoid(x, name='sigmoid'):

    net = tf.math.sigmoid(x, name=name)

    _log_hparams(classname='Sigmoid', layername=net.name, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net


def softmax(inputs, axis=None, name="softmax"):

    net = tf.nn.softmax(
        inputs,
        axis=axis,
        name=name,
    )

    _log_hparams(
        classname='Softmax', layername=net.name, axis=axis, out_shape=str(net.get_shape()), out_dtype=net.dtype
    )

    return net


def tanh(inputs, name='tanh'):

    net = tf.math.tanh(inputs, name=name)

    _log_hparams(classname='TanH', layername=net.name, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net
