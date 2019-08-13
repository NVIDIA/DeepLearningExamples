#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================


import tensorflow as tf

import model.layers
import model.blocks 

__all__ = ['bottleneck_block']


def bottleneck_block(
    inputs,
    depth,
    depth_bottleneck,
    stride,
    training=True,
    data_format='NCHW',
    conv2d_hparams=None,
    batch_norm_hparams=None,
    block_name="bottleneck_block"
):

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    if not isinstance(conv2d_hparams, tf.contrib.training.HParams):
        raise ValueError("The paramater `conv2d_hparams` is not of type `HParams`")

    if not isinstance(batch_norm_hparams, tf.contrib.training.HParams):
        raise ValueError("The paramater `conv2d_hparams` is not of type `HParams`")

    in_shape = inputs.get_shape()

    in_size = in_shape[1] if data_format == "NCHW" else in_shape[-1]

    with tf.variable_scope(block_name):

        with tf.variable_scope("shortcut"):
            if depth == in_size:

                if stride == 1:
                    shortcut = tf.identity(inputs)

                else:

                    shortcut = model.layers.average_pooling2d(
                        inputs,
                        pool_size=(1, 1),
                        strides=(stride, stride),
                        padding='valid',
                        data_format='channels_first' if data_format == 'NCHW' else 'channels_last',
                        name="average_pooling2d",
                    )

            else:

                shortcut = model.blocks.conv2d_block(
                    inputs,
                    n_channels=depth,
                    kernel_size=(1, 1),
                    strides=(stride, stride),
                    mode='SAME',
                    use_batch_norm=True,
                    activation=None,  # Applied at the end after addition with bottleneck
                    is_training=training,
                    data_format=data_format,
                    conv2d_hparams=conv2d_hparams,
                    batch_norm_hparams=batch_norm_hparams
                )

        bottleneck = model.blocks.conv2d_block(
            inputs,
            n_channels=depth_bottleneck,
            kernel_size=(1, 1),
            strides=(1, 1),
            mode='SAME',
            use_batch_norm=True,
            activation='relu',
            is_training=training,
            data_format=data_format,
            conv2d_hparams=conv2d_hparams,
            batch_norm_hparams=batch_norm_hparams,
            name='bottleneck_1'
        )

        bottleneck = model.blocks.conv2d_block(
            bottleneck,
            n_channels=depth_bottleneck,
            kernel_size=(3, 3),
            strides=(stride, stride),
            mode='SAME_RESNET',
            use_batch_norm=True,
            activation='relu',
            is_training=training,
            data_format=data_format,
            conv2d_hparams=conv2d_hparams,
            batch_norm_hparams=batch_norm_hparams,
            name='bottleneck_2'
        )

        bottleneck = model.blocks.conv2d_block(
            bottleneck,
            n_channels=depth,
            kernel_size=(1, 1),
            strides=(1, 1),
            mode='SAME',
            use_batch_norm=True,
            activation=None,  # Applied at the end after addition with shortcut
            is_training=training,
            data_format=data_format,
            conv2d_hparams=conv2d_hparams,
            batch_norm_hparams=batch_norm_hparams,
            name='bottleneck_3'
        )

        return model.layers.relu(shortcut + bottleneck, name='relu')
