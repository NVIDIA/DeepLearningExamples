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

from model import layers

from model.layers.utils import _log_hparams

__all__ = ['deconv2d']


def deconv2d(
    inputs,
    n_channels=8,
    kernel_size=(3, 3),
    padding='VALID',
    data_format='NHWC',
    use_bias=True,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer(),
    trainable=True,
    use_upscale_conv=True
):

    padding = padding.upper()  # Enforce capital letters for the padding mode

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    if padding not in ['SAME', 'VALID']:
        raise ValueError("Unknown padding: `%s` (accepted: ['SAME', 'VALID'])" % padding)

    with tf.variable_scope("deconv2d"):

        if use_upscale_conv:

            layer = layers.upscale_2d(
                inputs,
                size=(2, 2),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,  # [BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA]
                align_corners=True,
                is_scale=True,
                data_format=data_format
            )

            layer = layers.conv2d(
                layer,
                n_channels=n_channels,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding=padding,
                data_format=data_format,
                use_bias=use_bias,
                trainable=trainable,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer
            )

        else:

            input_shape = inputs.get_shape()

            layer = tf.layers.conv2d_transpose(
                inputs=inputs,
                filters=n_channels,
                kernel_size=kernel_size,
                strides=(2, 2),
                padding=padding,
                data_format='channels_first' if data_format == "NCHW" else "channels_last",
                use_bias=use_bias,
                trainable=trainable,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer
            )

            _log_hparams(
                classname='Conv2DTranspose',
                layername=layer.name,
                n_channels=n_channels,
                kernel_size=kernel_size,
                strides=(2, 2),
                padding=padding,
                data_format=data_format,
                use_bias=use_bias,
                trainable=trainable,
                input_shape=str(input_shape),
                out_shape=str(layer.get_shape()),
                out_dtype=layer.dtype
            )

    return layer
