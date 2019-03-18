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

__all__ = ['concat', 'flatten', 'reshape', 'squeeze', 'upscale_2d']


def concat(values, axis, name='concat'):

    net = tf.concat(values=values, axis=axis, name=name)

    _log_hparams(classname='Concat', layername=net.name, axis=axis, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net


def flatten(inputs, name='flatten'):

    net = tf.layers.flatten(inputs, name=name)

    _log_hparams(classname='Flatten', layername=net.name, out_shape=str(net.get_shape()), out_dtype=net.dtype)

    return net


def reshape(tensor, shape, name='reshape'):

    net = tf.reshape(tensor, shape=shape, name=name)

    _log_hparams(
        classname='Reshape', layername=net.name, shape=shape, out_shape=str(net.get_shape()), out_dtype=net.dtype
    )

    return net


def squeeze(tensor, axis, name='squeeze'):

    net = tf.squeeze(tensor, axis=axis, name=name)

    _log_hparams(
        classname='Squeeze', layername=net.name, axis=axis, out_shape=str(net.get_shape()), out_dtype=net.dtype
    )

    return net


def upscale_2d(inputs, size, is_scale=True, method=0, align_corners=True, data_format='NHWC', name='upsample2d_layer'):

    if not isinstance(size, (list, tuple)) and len(size) == 2:
        raise AssertionError()

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format received: `%s` (allowed: `NHWC`, `NCHW`)" % data_format)

    input_shape = inputs.get_shape()

    if len(inputs.get_shape()) == 3:
        if is_scale:
            size_h = size[0] * int(inputs.get_shape()[0])
            size_w = size[1] * int(inputs.get_shape()[1])
            _size = [size_h, size_w]
        else:
            _size = size

    elif len(inputs.get_shape()) == 4:
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])  # NCHW => NHWC

        if is_scale:
            size_h = size[0] * int(inputs.get_shape()[1])
            size_w = size[1] * int(inputs.get_shape()[2])
            _size = [size_h, size_w]
        else:
            _size = size

    else:
        raise Exception("Do not support shape %s" % str(inputs.get_shape()))

    with tf.variable_scope(name):
        net = tf.image.resize_images(inputs, size=_size, method=method, align_corners=align_corners)

    if data_format == 'NCHW' and len(inputs.get_shape()) == 4:
        net = tf.transpose(net, [0, 3, 1, 2])  # NHWC => NCHW

    _log_hparams(
        classname='Upscale2D',
        layername=net.name,
        size=size,
        is_scale=is_scale,
        method=method,
        align_corners=align_corners,
        data_format=data_format,
        input_shape=str(input_shape),
        out_shape=str(net.get_shape()),
        out_dtype=net.dtype
    )

    return net
