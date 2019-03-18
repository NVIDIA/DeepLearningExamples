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

from model.layers.utils import _log_hparams

from model.layers.activation import crelu
from model.layers.activation import elu
from model.layers.activation import leaky_relu
from model.layers.activation import prelu
from model.layers.activation import relu
from model.layers.activation import relu6
from model.layers.activation import selu
from model.layers.activation import sigmoid
from model.layers.activation import softmax
from model.layers.activation import tanh

from model.layers.conv2d import conv2d
from model.layers.deconv2d import deconv2d

from model.layers.dense import dense

from model.layers.drop_layers import dropout

from model.layers.math_ops import reduce_mean

from model.layers.normalization import batch_norm

from model.layers.padding import pad

from model.layers.pooling import average_pooling2d
from model.layers.pooling import max_pooling2d

from model.layers.array_ops import concat
from model.layers.array_ops import flatten
from model.layers.array_ops import reshape
from model.layers.array_ops import squeeze
from model.layers.array_ops import upscale_2d

__all__ = [

    # activation layers
    'crelu',
    'elu',
    'leaky_relu',
    'prelu',
    'relu',
    'relu6',
    'selu',
    'sigmoid',
    'softmax',
    'tanh',

    # array ops
    'concat',
    'flatten',
    'reshape',
    'squeeze',
    'upscale_2d',

    # conv layers
    'conv2d',

    # deconv layers
    'deconv2d',

    # dense layers
    'dense',

    # drop layers
    'dropout',

    # math_ops layers
    'reduce_mean',

    # normalization layers
    'batch_norm',

    # padding layers
    'pad',

    # pooling layers
    'average_pooling2d',
    'max_pooling2d',
]
