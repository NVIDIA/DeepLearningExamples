# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from typing import Any, Dict, Optional, Text, Tuple

from model.layers import get_batch_norm

__all__ = ['conv2d_block']

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_in',
        # Note: this is a truncated normal distribution
        'distribution': 'normal'
    }
}

def conv2d_block(inputs: tf.Tensor,
                 conv_filters: Optional[int],
                 config: dict,
                 kernel_size: Any = (1, 1),
                 strides: Any = (1, 1),
                 use_batch_norm: bool = True,
                 use_bias: bool = False,
                 activation: Any = None,
                 depthwise: bool = False,
                 name: Text = None):
  """A conv2d followed by batch norm and an activation."""
  batch_norm = get_batch_norm(config['batch_norm'])
  bn_momentum = config['bn_momentum']
  bn_epsilon = config['bn_epsilon']
  data_format = tf.keras.backend.image_data_format()
  weight_decay = config['weight_decay']

  name = name or ''

  # Collect args based on what kind of conv2d block is desired
  init_kwargs = {
      'kernel_size': kernel_size,
      'strides': strides,
      'use_bias': use_bias,
      'padding': 'same',
      'name': name + '_conv2d',
      'kernel_regularizer': tf.keras.regularizers.l2(weight_decay),
      'bias_regularizer': tf.keras.regularizers.l2(weight_decay),
  }
  CONV_KERNEL_INITIALIZER['config']['mode'] = config['weight_init']

  if depthwise:
    conv2d = tf.keras.layers.DepthwiseConv2D
    init_kwargs.update({'depthwise_initializer': CONV_KERNEL_INITIALIZER})
  else:
    conv2d = tf.keras.layers.Conv2D
    init_kwargs.update({'filters': conv_filters,
                        'kernel_initializer': CONV_KERNEL_INITIALIZER})

  x = conv2d(**init_kwargs)(inputs)

  if use_batch_norm:
    bn_axis = 1 if data_format == 'channels_first' else -1
    x = batch_norm(axis=bn_axis,
                   momentum=bn_momentum,
                   epsilon=bn_epsilon,
                   name=name + '_bn')(x)

  if activation is not None:
    x = tf.keras.layers.Activation(activation,
                                   name=name + '_activation')(x)
  return x