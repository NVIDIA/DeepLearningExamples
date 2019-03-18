# !/usr/bin/env python
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

__all__ = [
    "authorized_activation_fn",
    "activation_block",
]

authorized_activation_fn = ["relu", "leaky_relu", "prelu_shared", "prelu_not_shared", "selu", "crelu", "elu"]


def activation_block(inputs, act_fn, trainable=True, block_name='activation'):

    with tf.variable_scope(block_name):

        if act_fn == "relu":
            return layers.relu(inputs)

        if act_fn == "leaky_relu":
            return layers.leaky_relu(inputs, alpha=0.2)

        if act_fn == "prelu_shared":
            return layers.prelu(inputs, channel_shared=True, trainable=trainable)

        if act_fn == "prelu_not_shared":
            return layers.prelu(inputs, channel_shared=False, trainable=trainable)

        if act_fn == "selu":
            return layers.selu(inputs)

        if act_fn == "crelu":
            return layers.crelu(inputs)

        if act_fn == "elu":
            return layers.elu(inputs)

        raise ValueError("Unknown activation function: %s - Authorized: %s" % (act_fn, authorized_activation_fn))
