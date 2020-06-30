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

__all__ = ['relu', 'softmax', 'tanh', 'sigmoid']


def relu(inputs, name='relu'):

    net = tf.nn.relu(inputs, name=name)

    return net


def softmax(inputs, axis=None, name="softmax"):

    net = tf.nn.softmax(
        inputs,
        axis=axis,
        name=name,
    )

    return net


def tanh(inputs, name='tanh'):

    net = tf.math.tanh(inputs, name=name)

    return net

def sigmoid(inputs, name='sigmoid'):
    
    net = tf.math.sigmoid(inputs, name=name)
    
    return net
