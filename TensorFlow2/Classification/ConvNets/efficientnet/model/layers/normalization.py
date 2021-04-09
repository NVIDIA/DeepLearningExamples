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
# ==============================================================================
"""Common modeling utilities."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from typing import Text, Optional

from tensorflow.python.tpu import tpu_function

__all__ = ['get_batch_norm']

@tf.keras.utils.register_keras_serializable(package='Vision')
class TpuBatchNormalization(tf.keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused: Optional[bool] = False, **kwargs):
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t: tf.Tensor, num_shards_per_group: int):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    group_assignment = None
    if num_shards_per_group > 1:
      if num_shards % num_shards_per_group != 0:
        raise ValueError(
            'num_shards: %d mod shards_per_group: %d, should be 0' %
            (num_shards, num_shards_per_group))
      num_groups = num_shards // num_shards_per_group
      group_assignment = [[
          x for x in range(num_shards) if x // num_shards_per_group == y
      ] for y in range(num_groups)]
    return tf1.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
        num_shards_per_group, t.dtype)

  def _moments(self, inputs: tf.Tensor, reduction_axes: int, keep_dims: int):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
      num_shards_per_group = 1
    else:
      num_shards_per_group = max(8, num_shards // 8)
    if num_shards_per_group > 1:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = self._cross_replica_average(shard_mean, num_shards_per_group)
      group_mean_of_square = self._cross_replica_average(
          shard_mean_of_square, num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)

@tf.keras.utils.register_keras_serializable(package='Vision')
class SyncBatchNormalization(tf.keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    super(SyncBatchNormalization, self).__init__(**kwargs)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
    shard_mean, shard_variance = super(SyncBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = hvd.size()
    if num_shards > 1:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      shard_stack = tf.stack([shard_mean, shard_mean_of_square])
      group_mean, group_mean_of_square = tf.unstack(hvd.allreduce(shard_stack))
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)

  def call(self, *args, **kwargs):
    outputs = super(SyncBatchNormalization, self).call(*args, **kwargs)
    # A temporary hack for tf1 compatibility with keras batch norm.
    # for u in self.updates:
    #   tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    return outputs


def get_batch_norm(batch_norm_type: Text) -> tf.keras.layers.BatchNormalization:
  """A helper to create a batch normalization getter.

  Args:
    batch_norm_type: The type of batch normalization layer implementation. `tpu`
     will use `TpuBatchNormalization`.

  Returns:
    An instance of `tf.keras.layers.BatchNormalization`.
  """
  if batch_norm_type == 'tpu':
    return TpuBatchNormalization
  if batch_norm_type == 'syncbn':
    return SyncBatchNormalization

  return tf.keras.layers.BatchNormalization