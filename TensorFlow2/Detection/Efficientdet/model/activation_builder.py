# Copyright 2020 Google Research. All Rights Reserved.
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
"""Common utils."""
from typing import Text
import tensorflow as tf


def srelu_fn(x):
  """Smooth relu: a smooth version of relu."""
  with tf.name_scope('srelu'):
    beta = tf.Variable(20.0, name='srelu_beta', dtype=tf.float32)**2
    beta = tf.cast(beta**2, x.dtype)
    safe_log = tf.math.log(tf.where(x > 0., beta * x + 1., tf.ones_like(x)))
    return tf.where((x > 0.), x - (1. / beta) * safe_log, tf.zeros_like(x))


def activation_fn(features: tf.Tensor, act_type: Text):
  """Customized non-linear activation type."""
  if act_type in ('silu', 'swish'):
    return tf.nn.swish(features)
  elif act_type == 'swish_native':
    return features * tf.sigmoid(features)
  elif act_type == 'hswish':
    return features * tf.nn.relu6(features + 3) / 6
  elif act_type == 'relu':
    return tf.nn.relu(features)
  elif act_type == 'relu6':
    return tf.nn.relu6(features)
  elif act_type == 'mish':
    return features * tf.math.tanh(tf.math.softplus(features))
  elif act_type == 'identity':
    return tf.identity(features)
  elif act_type == 'srelu':
    return srelu_fn(features)
  else:
    raise ValueError('Unsupported act_type {}'.format(act_type))