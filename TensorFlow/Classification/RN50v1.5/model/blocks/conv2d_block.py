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
    name='conv2d'
):

    if not isinstance(conv2d_hparams, tf.contrib.training.HParams):
        raise ValueError("The paramater `conv2d_hparams` is not of type `HParams`")

    if not isinstance(batch_norm_hparams, tf.contrib.training.HParams) and use_batch_norm:
        raise ValueError("The paramater `conv2d_hparams` is not of type `HParams`")

    with tf.variable_scope(name):

        if mode != 'SAME_RESNET':
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
                bias_initializer=conv2d_hparams.bias_initializer,
            )

        else:  # Special padding mode for ResNet models
            if strides == (1, 1):

                net = layers.conv2d(
                    inputs,
                    n_channels=n_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='SAME',
                    data_format=data_format,
                    use_bias=not use_batch_norm,
                    trainable=is_training,
                    kernel_initializer=conv2d_hparams.kernel_initializer,
                    bias_initializer=conv2d_hparams.bias_initializer,
                )

            else:
                rate = 1  # Unused (for 'a trous' convolutions)

                kernel_height_effective = kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)

                pad_h_beg = (kernel_height_effective - 1) // 2
                pad_h_end = kernel_height_effective - 1 - pad_h_beg

                kernel_width_effective = kernel_size[1] + (kernel_size[1] - 1) * (rate - 1)

                pad_w_beg = (kernel_width_effective - 1) // 2
                pad_w_end = kernel_width_effective - 1 - pad_w_beg

                padding = [[0, 0], [pad_h_beg, pad_h_end], [pad_w_beg, pad_w_end], [0, 0]]

                if data_format == 'NCHW':
                    padding = [padding[0], padding[3], padding[1], padding[2]]

                padded_inputs = tf.pad(inputs, padding)

                net = layers.conv2d(
                    padded_inputs,  # inputs,
                    n_channels=n_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='VALID',
                    data_format=data_format,
                    use_bias=not use_batch_norm,
                    trainable=is_training,
                    kernel_initializer=conv2d_hparams.kernel_initializer,
                    bias_initializer=conv2d_hparams.bias_initializer,
                )

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
