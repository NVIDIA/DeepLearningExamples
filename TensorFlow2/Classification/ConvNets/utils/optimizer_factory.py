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
"""Optimizer factory for vision tasks."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, Dict, Text, List
from tensorflow import keras
from tensorflow_addons.optimizers import MovingAverage
# pylint: disable=protected-access

from utils import learning_rate


def fetch_optimizer(model,opt_type) -> tf.keras.optimizers.Optimizer:
  """Get the base optimizer used by the current model."""
  
  # this is the case where our target optimizer is not wrapped by any other optimizer(s)
  if isinstance(model.optimizer,opt_type):
    return model.optimizer
  
  # Dive into nested optimizer object until we reach the target opt
  opt = model.optimizer
  while hasattr(opt, '_optimizer'):
    opt = opt._optimizer
    if isinstance(opt,opt_type):
      return opt 
  raise TypeError(f'Failed to find {opt_type} in the nested optimizer object')

# Inspired from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py
class GradientAccumulator(object):
  """Distribution strategies-aware gradient accumulation utility."""

  def __init__(self):
      """Initializes the accumulator."""
      self._gradients = []
      self._accum_steps = tf.Variable(
          initial_value=0, dtype=tf.int64, trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
      )

  @property
  def step(self):
      """Number of accumulated steps."""
      return self._accum_steps.value()

  @property
  def gradients(self):
      """The accumulated gradients."""
      return list(
          gradient.value() if gradient is not None else gradient for gradient in self._get_replica_gradients()
      )

  def __call__(self, gradients):
      """Accumulates :obj:`gradients`."""
      if not self._gradients:
          self._gradients.extend(
              [
                  tf.Variable(tf.zeros_like(gradient), trainable=False) if gradient is not None else gradient
                  for gradient in gradients
              ]
          )

      if len(gradients) != len(self._gradients):
          raise ValueError("Expected %s gradients, but got %d" % (len(self._gradients), len(gradients)))

      for accum_gradient, gradient in zip(self._get_replica_gradients(), gradients):
          if accum_gradient is not None and gradient is not None:
              accum_gradient.assign_add(gradient)

      self._accum_steps.assign_add(1)

  def reset(self):
      """Resets the accumulated gradients."""
      if self._gradients:
          self._accum_steps.assign(0)

      for gradient in self._get_replica_gradients():
          if gradient is not None:
              gradient.assign(tf.zeros_like(gradient))

  def normalize(self):
      """Normalizes the accumulated gradients."""
      for gradient in self._get_replica_gradients():
          if gradient is not None:
              gradient.assign(gradient*tf.cast(1/self._accum_steps, gradient.dtype))
              
  def _get_replica_gradients(self):
      if tf.distribute.has_strategy():
          # In a replica context, we want to accumulate gradients on each replica
          # without synchronization, so we directly assign the value of the
          # current replica.
          replica_context = tf.distribute.get_replica_context()

          if replica_context is None or tf.distribute.get_strategy().num_replicas_in_sync == 1:
              return self._gradients

          return (
              gradient.device_map.select_for_current_replica(gradient.values, replica_context)
              for gradient in self._gradients
              if gradient is not None
          )
      else:
          return self._gradients


class HvdMovingAverage(MovingAverage):
    
  def swap_weights(self):
    """Swap the average and moving weights. 
    The original function in the parent class assumes a cross replica
    context, which fails for single GPU training. It also failed in the case of 
    multi-GPU training with Horovod.
    """
    self._swap_weights()
    
  def _create_slots(self, var_list):
    """[summary]
    The original function in the parent class, in addition to calling
    _create_slots() of the base optimizer, reassigns trainable tensors to
    self._average_weights and self._model_weights, which has the effect of
    removing non-trainable tensors (e.g., moving means and variances) from EMA. 
    By overriding it, we simply keep the part that calls _create_slots of the base
    optimizer. To make up for the removed part of the code, we call shadow_copy, which
    assigns both trainable and non-trainable tensors to self._average_weights and 
    self._model_weights.
    Args:
        var_list ([type]): [description]
    """
    self._optimizer._create_slots(var_list=var_list)

        
  def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
      self._optimizer._iterations = self.iterations
      result = super().apply_gradients(grads_and_vars, name)
      # update EMA weights after the weights are updated
      self.update_average(self._optimizer.iterations)
      return result

  def _resource_apply_dense(self, grad, var):
    """[summary]
    We must override this function, eliminating the part that performs
    EMA updates for trainable variables. The reasons is that we use our custom 
    self.update_average(), called in apply_gradients, which performs EMA updates
    for both trainable and non-trainable variables. If we don't override this 
    function, in each iteration, EMA of trainable variables get updated twice 
    (once here and once in apply_gradient) while EMA of non-trainable variables get
    updated only once in apply_gradients.
    """
    return self._optimizer._resource_apply_dense(grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
    """[summary]
    We must override this function, eliminating the part that performs
    EMA updates for trainable variables. The reasons is that we use our custom 
    self.update_average(), called in apply_gradients, which performs EMA updates
    for both trainable and non-trainable variables. If we don't override this 
    function, in each iteration, EMA of trainable variables get updated twice 
    (once here and once in apply_gradient) while EMA of non-trainable variables get
    updated only once in apply_gradients.
    """
    return self._optimizer._resource_apply_sparse(grad, var, indices)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
    """[summary]
    We must override this function, eliminating the part that performs
    EMA updates for trainable variables. The reasons is that we use our custom 
    self.update_average(), called in apply_gradients, which performs EMA updates
    for both trainable and non-trainable variables. If we don't override this 
    function, in each iteration, EMA of trainable variables get updated twice 
    (once here and once in apply_gradient) while EMA of non-trainable variables get
    updated only once in apply_gradients.
    """

    return self._optimizer._resource_apply_sparse_duplicate_indices(
        grad, var, indices)

  @tf.function
  def update_average(self, step: tf.Tensor):
    step = tf.cast(step, tf.float32)
    average_decay = self._get_hyper("average_decay", tf.dtypes.float32)
    if step < self._start_step:
      decay = tf.constant(0., tf.float32)
    elif self._dynamic_decay:
      decay = step - self._start_step
      decay = tf.minimum(average_decay, (1. + decay) / (10. + decay))
    else:
      decay = average_decay

    def _apply_moving(v_moving, v_normal):
      diff = v_moving - v_normal
      v_moving.assign_sub(tf.cast(1. - decay, v_moving.dtype) * diff)
      return v_moving

    def _update(strategy, v_moving_and_v_normal):
      for v_moving, v_normal in v_moving_and_v_normal:
        strategy.extended.update(v_moving, _apply_moving, args=(v_normal,))

    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(_update, args=(zip(self._average_weights,
                                            self._model_weights),))



  @classmethod
  def from_config(cls, config, custom_objects=None):
    optimizer = tf.keras.optimizers.deserialize(
        config.pop('optimizer'),
        custom_objects=custom_objects,
    )
    # For some reason, it is necessary to pass the optimizer as a keyword arg
    return cls(optimizer=optimizer, **config)
    
    
def build_optimizer(
    optimizer_name: Text,
    base_learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
    params: Dict[Text, Any]):
  """Build the optimizer based on name.

  Args:
    optimizer_name: String representation of the optimizer name. Examples:
      sgd, momentum, rmsprop.
    base_learning_rate: `tf.keras.optimizers.schedules.LearningRateSchedule`
      base learning rate.
    params: String -> Any dictionary representing the optimizer params.
      This should contain optimizer specific parameters such as
      `base_learning_rate`, `decay`, etc.

  Returns:
    A tf.keras.Optimizer.

  Raises:
    ValueError if the provided optimizer_name is not supported.

  """
  optimizer_name = optimizer_name.lower()

  if optimizer_name == 'sgd':
    nesterov = params.get('nesterov', False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_learning_rate,
                                        nesterov=nesterov)
  elif optimizer_name == 'momentum':
    nesterov = params.get('nesterov', False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_learning_rate,
                                        momentum=params['momentum'],
                                        nesterov=nesterov)
  elif optimizer_name == 'rmsprop':
    rho = params.get('decay', None) or params.get('rho', 0.9)
    momentum = params.get('momentum', 0.9)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate,
                                            rho=rho,
                                            momentum=momentum,
                                            epsilon=epsilon)
  elif optimizer_name == 'adam':
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         epsilon=epsilon)
  elif optimizer_name == 'adamw':
    weight_decay = params.get('weight_decay', 0.01)
    beta_1 = params.get('beta_1', 0.9)
    beta_2 = params.get('beta_2', 0.999)
    epsilon = params.get('epsilon', 1e-07)
    optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay,
                                     learning_rate=base_learning_rate,
                                     beta_1=beta_1,
                                     beta_2=beta_2,
                                     epsilon=epsilon)
  else:
    raise ValueError('Unknown optimizer %s' % optimizer_name)

  if params.get('lookahead', None):
    optimizer = tfa.optimizers.Lookahead(optimizer)

  # Moving average should be applied last, as it's applied at test time
  moving_average_decay = params.get('moving_average_decay', 0.)
  if moving_average_decay is not None and moving_average_decay > 0.:
    optimizer = HvdMovingAverage(# tfa.optimizers.MovingAverage
        optimizer,
        average_decay=moving_average_decay,
        dynamic_decay=True)
  return optimizer


def build_learning_rate(params: Dict[Text, Any],
                        batch_size: int = None,
                        train_steps: int = None,
                        max_epochs: int = None):
  """Build the learning rate given the provided configuration."""
  decay_type = params['name']
  base_lr = params['initial_lr']
  decay_rate = params['decay_rate']
  if params['decay_epochs'] is not None:
    decay_steps = params['decay_epochs'] * train_steps
  else:
    decay_steps = 0
  if params['warmup_epochs'] is not None:
    warmup_steps = params['warmup_epochs'] * train_steps
  else:
    warmup_steps = 0

  lr_multiplier = params['scale_by_batch_size']

  if lr_multiplier and lr_multiplier > 0:
    # Scale the learning rate based on the batch size and a multiplier
    base_lr *= lr_multiplier * batch_size

  if decay_type == 'exponential':
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=params['staircase'])
  elif decay_type == 'piecewise_constant_with_warmup':
    lr = learning_rate.PiecewiseConstantDecayWithWarmup(
        batch_size=batch_size,
        epoch_size=params['examples_per_epoch'],
        warmup_epochs=params['warmup_epochs'],
        boundaries=params['boundaries'],
        multipliers=params['multipliers'])
  elif decay_type == 'cosine':
    decay_steps = (max_epochs - params['warmup_epochs']) * train_steps
    lr = tf.keras.experimental.CosineDecay(
        initial_learning_rate=base_lr,
        decay_steps=decay_steps,
        alpha=0.0
    )
  elif decay_type == 'linearcosine':
    decay_steps = (max_epochs - params['warmup_epochs']) * train_steps
    lr = tf.keras.experimental.NoisyLinearCosineDecay(
        initial_learning_rate=base_lr,
        decay_steps=decay_steps,
        initial_variance=0.5, 
        variance_decay=0.55,
        num_periods=0.5, alpha=0.0, beta=0.001
    )
  if warmup_steps > 0:
    if decay_type != 'piecewise_constant_with_warmup':
      lr = learning_rate.WarmupDecaySchedule(lr, warmup_steps)
  return lr

