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
from typing import Text, Union
import tensorflow as tf
import horovod.tensorflow.keras as hvd

from utils.horovod_utils import get_world_size
from model import activation_builder




class BatchNormalization(tf.keras.layers.BatchNormalization):
  """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

  def __init__(self, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    super().__init__(**kwargs)


def batch_norm_class(is_training=True):
  if is_training:
    # TODO(fsx950223): use SyncBatchNorm after TF bug is fixed (incorrect nccl
    # all_reduce). See https://github.com/tensorflow/tensorflow/issues/41980
    return BatchNormalization
  else:
    return BatchNormalization


def batch_normalization(inputs, training=False, **kwargs):
  """A wrapper for TpuBatchNormalization."""
  bn_layer = batch_norm_class(training)(**kwargs)
  return bn_layer(inputs, training=training)


def batch_norm_act(inputs,
                   is_training_bn: bool,
                   act_type: Union[Text, None],
                   init_zero: bool = False,
                   data_format: Text = 'channels_last',
                   momentum: float = 0.99,
                   epsilon: float = 1e-3,
                   name: Text = None):
  """Performs a batch normalization followed by a non-linear activation.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    act_type: non-linear relu function type. If None, omits the relu operation.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      training=is_training_bn,
      gamma_initializer=gamma_initializer,
      name=name)

  if act_type:
    inputs = activation_builder.activation_fn(inputs, act_type)
  return inputs
