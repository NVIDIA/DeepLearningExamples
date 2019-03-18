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

__all__ = ['average_pooling2d', 'max_pooling2d']


def average_pooling2d(inputs, pool_size=(2, 2), strides=None, padding='valid', data_format=None, name="avg_pooling2d"):

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    if padding.lower() not in ['same', 'valid']:
        raise ValueError("Unknown padding: `%s` (accepted: ['same', 'valid'])" % padding)

    net = tf.layers.average_pooling2d(
        inputs,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format='channels_first' if data_format == 'NCHW' else 'channels_last',
        name=name
    )

    return net


def max_pooling2d(inputs, pool_size=(2, 2), strides=None, padding='valid', data_format=None, name="max_pooling2d"):

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    if padding.lower() not in ['same', 'valid']:
        raise ValueError("Unknown padding: `%s` (accepted: ['same', 'valid'])" % padding)

    net = tf.layers.max_pooling2d(
        inputs,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format='channels_first' if data_format == 'NCHW' else 'channels_last',
        name=name
    )

    return net
