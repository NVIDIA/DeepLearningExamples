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

import inspect

import tensorflow as tf

from model.layers import _log_hparams

__all__ = ['batch_norm']


def batch_norm(
    inputs,
    decay=0.999,
    epsilon=0.001,
    scale=False,
    center=True,
    is_training=True,
    data_format='NHWC',
    param_initializers=None
):
    """Adds a Batch Normalization layer."""

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    if param_initializers is not None:

        for key, initializer in param_initializers.items():

            if key not in ['beta', 'gamma', 'moving_mean', 'moving_variance']:
                raise ValueError("Unknown key received: `%s`" % key)

            if inspect.isclass(initializer):
                initializer = initializer()
                setattr(param_initializers, key, initializer)

            if initializer.__class__.__module__ != 'tensorflow.python.ops.init_ops':
                raise ValueError("The object `%s` is not a Tensor initializer" % str(initializer))

    input_shape = inputs.get_shape()
    input_rank = input_shape.ndims
    input_channels = input_shape[1]

    if input_rank == 2:

        if data_format == 'NCHW':
            new_shape = [-1, input_channels, 1, 1]
        else:
            new_shape = [-1, 1, 1, input_channels]

        inputs = tf.reshape(inputs, new_shape)

    net = tf.contrib.layers.batch_norm(
        inputs,
        decay=decay,
        scale=scale,
        epsilon=epsilon,
        is_training=is_training,
        trainable=is_training,
        fused=True,
        data_format=data_format,
        center=center,
        param_initializers=param_initializers
    )

    if input_rank == 2:
        net = tf.reshape(net, [-1, input_channels])

    _log_hparams(
        classname='BatchNorm',
        layername=net.name,
        data_format=data_format,
        is_training=is_training,
        decay=decay,
        epsilon=epsilon,
        scale=scale,
        center=center,
        fused=True,
        out_shape=str(net.get_shape()),
        out_dtype=net.dtype
    )

    return net