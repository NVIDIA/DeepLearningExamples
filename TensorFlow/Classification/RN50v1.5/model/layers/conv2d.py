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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = ['conv2d']


def conv2d(
    inputs,
    n_channels=8,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='VALID',
    data_format='NHWC',
    dilation_rate=(1, 1),
    use_bias=True,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer(),
    trainable=True,
    name=None
):

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    if padding.upper() not in ['SAME', 'VALID']:
        raise ValueError("Unknown padding: `%s` (accepted: ['SAME', 'VALID'])" % padding.upper())

    net = tf.layers.conv2d(
        inputs,
        filters=n_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        data_format='channels_last' if data_format == 'NHWC' else 'channels_first',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        activation=None,
        name=name
    )
    
    return net

