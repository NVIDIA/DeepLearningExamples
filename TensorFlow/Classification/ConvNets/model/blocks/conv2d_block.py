#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf

from model import layers

__all__ = ['conv2d_block']


def conv2d_block(
    inputs,
    n_channels,
    kernel_size=(3, 3),
    strides=(2, 2),
    mode='SAME',
    use_batch_norm=True,
    activation='relu',
    is_training=True,
    data_format='NHWC',
    conv2d_hparams=None,
    batch_norm_hparams=None,
    name='conv2d',
    cardinality=1,
):

    if not isinstance(conv2d_hparams, tf.contrib.training.HParams):
        raise ValueError("The paramater `conv2d_hparams` is not of type `HParams`")

    if not isinstance(batch_norm_hparams, tf.contrib.training.HParams) and use_batch_norm:
        raise ValueError("The paramater `conv2d_hparams` is not of type `HParams`")

    with tf.variable_scope(name):
        if cardinality == 1:
            net = layers.conv2d(
                inputs,
                n_channels=n_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=mode,
                data_format=data_format,
                use_bias=not use_batch_norm,
                trainable=is_training,
                kernel_initializer=conv2d_hparams.kernel_initializer,
                bias_initializer=conv2d_hparams.bias_initializer)
        else:
            group_filter = tf.get_variable(
                name=name + 'group_filter',
                shape=[3, 3, n_channels // cardinality, n_channels],
                trainable=is_training,
                dtype=tf.float32)
            net = tf.nn.conv2d(inputs,
                               group_filter,
                               strides=strides,
                               padding='SAME',
                               data_format=data_format)
        if use_batch_norm:
            net = layers.batch_norm(
                net,
                decay=batch_norm_hparams.decay,
                epsilon=batch_norm_hparams.epsilon,
                scale=batch_norm_hparams.scale,
                center=batch_norm_hparams.center,
                is_training=is_training,
                data_format=data_format,
                param_initializers=batch_norm_hparams.param_initializers
            )

        if activation == 'relu':
            net = layers.relu(net, name='relu')

        elif activation == 'tanh':
            net = layers.tanh(net, name='tanh')

        elif activation != 'linear' and activation is not None:
            raise KeyError('Invalid activation type: `%s`' % activation)

        return net
