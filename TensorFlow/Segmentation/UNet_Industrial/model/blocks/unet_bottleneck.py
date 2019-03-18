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

__all__ = ["bottleneck_unet_block"]


def bottleneck_unet_block(
    inputs, filters, data_format='NCHW', is_training=True, conv2d_hparams=None, block_name='bottleneck_block'
):

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

        net = layers.deconv2d(
            net,
            n_channels=filters / 2,
            kernel_size=(2, 2),
            padding='same',
            data_format=data_format,
            use_bias=True,
            trainable=is_training,
            kernel_initializer=conv2d_hparams.kernel_initializer,
            bias_initializer=conv2d_hparams.bias_initializer,
        )

        net = blocks.activation_block(
            inputs=net, act_fn=conv2d_hparams.activation_fn, trainable=is_training, block_name='act3'
        )

        return net
