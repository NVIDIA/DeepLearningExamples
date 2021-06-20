#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================


import tensorflow as tf

from model import layers
from model import blocks

__all__ = ['squeeze_excitation_layer']

def squeeze_excitation_layer(
    inputs,
    ratio,
    training=True,
    data_format='NCHW',
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer(),
    name="squeeze_excitation_layer"
):

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    in_shape = inputs.get_shape()

    num_channels = in_shape[1] if data_format == "NCHW" else in_shape[-1]

    with tf.variable_scope(name):

        net = inputs
        
        # squeeze
        squeeze = layers.reduce_mean(
                    net, 
                    keepdims=False, 
                    data_format=data_format,
                    name='squeeze_spatial_mean'
                )

        # fc + relu
        excitation = layers.dense(
                    inputs=squeeze,
                    units=num_channels // ratio,
                    use_bias=True,
                    trainable=training,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer
                )
        excitation = layers.relu(excitation)
        
        # fc + sigmoid
        excitation = layers.dense(
                    inputs=excitation,
                    units=num_channels,
                    use_bias=True,
                    trainable=training,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer
                )
        excitation = layers.sigmoid(excitation)
        
        out_shape = [-1, num_channels, 1, 1] if data_format == "NCHW" else [-1, 1, 1, num_channels] 
        
        excitation = tf.reshape(excitation, out_shape)
        
        net = net * excitation
        
        return net
