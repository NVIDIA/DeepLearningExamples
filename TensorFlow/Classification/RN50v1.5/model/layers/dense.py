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

import tensorflow as tf

__all__ = ['dense']


def dense(
    inputs,
    units,
    use_bias=True,
    trainable=True,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer()
):

    net = tf.layers.dense(
        inputs,
        units=units,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable
    )

    return net
