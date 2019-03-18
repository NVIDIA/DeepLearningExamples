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

__all__ = ['reduce_mean']


def reduce_mean(inputs, keepdims=None, data_format='channels_last', name='spatial_mean'):

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    axes = [1, 2] if data_format == 'NHWC' else [2, 3]

    net = tf.math.reduce_mean(inputs, axis=axes, keepdims=keepdims, name=name)

    _log_hparams(
        classname='ReduceMean',
        layername=net.name,
        axis=axes,
        keepdims=keepdims,
        out_shape=str(net.get_shape()),
        out_dtype=net.dtype
    )

    return net
