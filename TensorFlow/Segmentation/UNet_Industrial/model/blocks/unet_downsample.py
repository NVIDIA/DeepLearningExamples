# !/usr/bin/env python
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

from model import layers
from model import blocks

__all__ = ["downsample_unet_block"]


def downsample_unet_block(
    inputs, filters, data_format='NCHW', is_training=True, conv2d_hparams=None, block_name='downsample_block'
):

    if not isinstance(conv2d_hparams, tf.contrib.training.HParams):
        raise ValueError("The paramater `conv2d_hparams` is not of type `HParams`")

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    with tf.variable_scope(block_name):

        net = layers.conv2d(
            inputs,
            n_channels=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            use_bias=True,
            trainable=is_training,
            kernel_initializer=conv2d_hparams.kernel_initializer,
            bias_initializer=conv2d_hparams.bias_initializer,
        )

        net = blocks.activation_block(
            inputs=net, act_fn=conv2d_hparams.activation_fn, trainable=is_training, block_name='act1'
        )

        net = layers.conv2d(
            net,
            n_channels=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            data_format=data_format,
            use_bias=True,
            trainable=is_training,
            kernel_initializer=conv2d_hparams.kernel_initializer,
            bias_initializer=conv2d_hparams.bias_initializer,
        )

        net = blocks.activation_block(
            inputs=net, act_fn=conv2d_hparams.activation_fn, trainable=is_training, block_name='act2'
        )

        outputs = layers.max_pooling2d(
            inputs=net,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            data_format=data_format,
            name="max_pooling2d"
        )

        return outputs, net
