# Lint as: python3
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
"""Learning rate utilities for vision tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, List, Mapping

import tensorflow as tf

BASE_LEARNING_RATE = 0.1

__all__ = [ 'WarmupDecaySchedule', 'PiecewiseConstantDecayWithWarmup' ]

@tf.keras.utils.register_keras_serializable(package='Custom')
class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""

  def __init__(
      self,
      lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
      warmup_steps: int,
      **kwargs):
    """Add warmup decay to a learning rate schedule.

    Args:
      lr_schedule: base learning rate scheduler
      warmup_steps: number of warmup steps

    """
    super(WarmupDecaySchedule, self).__init__()
    self._lr_schedule = lr_schedule
    self._warmup_steps = warmup_steps


  def __call__(self, step: int):
    lr = self._lr_schedule(step)
    if self._warmup_steps:
      step_decay = step - self._warmup_steps
      lr = self._lr_schedule(step_decay)
      initial_learning_rate = tf.convert_to_tensor(
          self._lr_schedule.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      global_step_recomp = tf.cast(step, dtype)
      warmup_steps = tf.cast(self._warmup_steps, dtype)
      warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
      lr = tf.cond(global_step_recomp < warmup_steps,
                   lambda: warmup_lr,
                   lambda: lr)
    return lr

  def get_config(self) -> Mapping[str, Any]:
    config = self._lr_schedule.get_config()
    config.update({
        "warmup_steps": self._warmup_steps,
    })
    config.update({
        "lr_schedule": self._lr_schedule,
    })
    return config



# TODO(b/149030439) - refactor this with
# tf.keras.optimizers.schedules.PiecewiseConstantDecay + WarmupDecaySchedule.
class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup schedule."""

  def __init__(self,
               batch_size: int,
               epoch_size: int,
               warmup_epochs: int,
               boundaries: List[int],
               multipliers: List[float]):
    """Piecewise constant decay with warmup.

    Args:
      batch_size: The training batch size used in the experiment.
      epoch_size: The size of an epoch, or the number of examples in an epoch.
      warmup_epochs: The number of warmup epochs to apply.
      boundaries: The list of floats with strictly increasing entries.
      multipliers: The list of multipliers/learning rates to use for the
        piecewise portion. The length must be 1 less than that of boundaries.

    """
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    if len(boundaries) != len(multipliers) - 1:
      raise ValueError("The length of boundaries must be 1 less than the "
                       "length of multipliers")

    base_lr_batch_size = 256
    steps_per_epoch = epoch_size // batch_size

    self._rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
    self._step_boundaries = [float(steps_per_epoch) * x for x in boundaries]
    self._lr_values = [self._rescaled_lr * m for m in multipliers]
    self._warmup_steps = warmup_epochs * steps_per_epoch

  def __call__(self, step: int):
    """Compute learning rate at given step."""
    def warmup_lr():
      return self._rescaled_lr * (
          step / tf.cast(self._warmup_steps, tf.float32))
    def piecewise_lr():
      return tf.compat.v1.train.piecewise_constant(
          tf.cast(step, tf.float32), self._step_boundaries, self._lr_values)
    return tf.cond(step < self._warmup_steps, warmup_lr, piecewise_lr)

  def get_config(self) -> Mapping[str, Any]:
    return {
        "rescaled_lr": self._rescaled_lr,
        "step_boundaries": self._step_boundaries,
        "lr_values": self._lr_values,
        "warmup_steps": self._warmup_steps,
    }
