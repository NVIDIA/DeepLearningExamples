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
"""Learning rate related utils."""
import math
from absl import logging
from typing import Any, Mapping
import tensorflow as tf


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule."""
  steps_per_epoch = params['steps_per_epoch']
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['total_steps'] = int(params['num_epochs'] * steps_per_epoch)


@tf.keras.utils.register_keras_serializable(package='Custom')
class CosineLrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Cosine learning rate schedule."""

  def __init__(self, base_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, total_steps: int):
    """Build a CosineLrSchedule.

    Args:
      base_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      total_steps: `int`, Total train steps.
    """
    super(CosineLrSchedule, self).__init__()
    logging.info('LR schedule method: cosine')
    self.base_lr = base_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.base_lr - self.lr_warmup_init)))
    cosine_lr = 0.5 * self.base_lr * (
        1 + tf.cos(math.pi * (tf.cast(step, tf.float32) - self.lr_warmup_step) / self.decay_steps))
    return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)

  def get_config(self) -> Mapping[str, Any]:
    return {
        "base_lr": self.base_lr,
        "lr_warmup_init": self.lr_warmup_init,
        "lr_warmup_step": self.lr_warmup_step,
    }


def learning_rate_schedule(params):
  """Learning rate schedule based on global step."""
  update_learning_rate_schedule_parameters(params)
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'cosine':
    return CosineLrSchedule(params['learning_rate'],
                            params['lr_warmup_init'], params['lr_warmup_step'],
                            params['total_steps'])

  raise ValueError('unknown lr_decay_method: {}'.format(lr_decay_method))
