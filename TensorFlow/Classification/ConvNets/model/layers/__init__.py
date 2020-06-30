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

from model.layers.activation import relu
from model.layers.activation import softmax
from model.layers.activation import tanh
from model.layers.activation import sigmoid

from model.layers.conv2d import conv2d

from model.layers.dense import dense

from model.layers.math_ops import reduce_mean

from model.layers.normalization import batch_norm

from model.layers.padding import pad

from model.layers.pooling import average_pooling2d
from model.layers.pooling import max_pooling2d
from model.layers.squeeze_excitation_layer import squeeze_excitation_layer

__all__ = [

    # activation layers
    'relu',
    'softmax',
    'tanh',
    'sigmoid',

    # conv layers
    'conv2d',

    # dense layers
    'dense',

    # math_ops layers
    'reduce_mean',

    # normalization layers
    'batch_norm',

    # padding layers
    'pad',

    # pooling layers
    'average_pooling2d',
    'max_pooling2d',

    'squeeze_excitation_layer'
]
