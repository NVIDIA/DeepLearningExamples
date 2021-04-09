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
"""Common modules for callbacks."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
from typing import Any, List, MutableMapping, Text
import tensorflow as tf
from tensorflow import keras

from utils import optimizer_factory
import horovod.tensorflow as hvd
import time


def get_callbacks(model_checkpoint: bool = True,
                  include_tensorboard: bool = True,
                  time_history: bool = True,
                  track_lr: bool = True,
                  write_model_weights: bool = True,
                  initial_step: int = 0,
                  batch_size: int = 0,
                  log_steps: int = 100,
                  model_dir: str = None,
                  save_checkpoint_freq: int = 0,
                  logger = None) -> List[tf.keras.callbacks.Callback]:
  """Get all callbacks."""
  model_dir = model_dir or ''
  callbacks = []
  if model_checkpoint and hvd.rank() == 0:
    ckpt_full_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        ckpt_full_path, save_weights_only=True, verbose=1, save_freq=save_checkpoint_freq))
  if time_history and logger is not None and hvd.rank() == 0:
    callbacks.append(
        TimeHistory(
            batch_size,
            log_steps,
            logdir=model_dir if include_tensorboard else None,
            logger=logger))
  if include_tensorboard:
    callbacks.append(
        CustomTensorBoard(
            log_dir=model_dir,
            track_lr=track_lr,
            initial_step=initial_step,
            write_images=write_model_weights))
  return callbacks


def get_scalar_from_tensor(t: tf.Tensor) -> int:
  """Utility function to convert a Tensor to a scalar."""
  t = tf.keras.backend.get_value(t)
  if callable(t):
    return t()
  else:
    return t


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
  """A customized TensorBoard callback that tracks additional datapoints.

  Metrics tracked:
  - Global learning rate

  Attributes:
    log_dir: the path of the directory where to save the log files to be parsed
      by TensorBoard.
    track_lr: `bool`, whether or not to track the global learning rate.
    initial_step: the initial step, used for preemption recovery.
    **kwargs: Additional arguments for backwards compatibility. Possible key is
      `period`.
  """

  # TODO(b/146499062): track params, flops, log lr, l2 loss,
  # classification loss

  def __init__(self,
               log_dir: str,
               track_lr: bool = False,
               initial_step: int = 0,
               **kwargs):
    super(CustomTensorBoard, self).__init__(log_dir=log_dir, **kwargs)
    self.step = initial_step
    self._track_lr = track_lr

  def on_batch_begin(self,
                     epoch: int,
                     logs: MutableMapping[str, Any] = None) -> None:
    self.step += 1
    if logs is None:
      logs = {}
    logs.update(self._calculate_metrics())
    super(CustomTensorBoard, self).on_batch_begin(epoch, logs)

  def on_epoch_begin(self,
                     epoch: int,
                     logs: MutableMapping[str, Any] = None) -> None:
    if logs is None:
      logs = {}
    metrics = self._calculate_metrics()
    logs.update(metrics)
    super(CustomTensorBoard, self).on_epoch_begin(epoch, logs)

  def on_epoch_end(self,
                   epoch: int,
                   logs: MutableMapping[str, Any] = None) -> None:
    if logs is None:
      logs = {}
    metrics = self._calculate_metrics()
    logs.update(metrics)
    super(CustomTensorBoard, self).on_epoch_end(epoch, logs)

  def _calculate_metrics(self) -> MutableMapping[str, Any]:
    logs = {}
    # TODO(b/149030439): disable LR reporting.
    if self._track_lr:
      logs['learning_rate'] = self._calculate_lr()
    return logs

  def _calculate_lr(self) -> int:
    """Calculates the learning rate given the current step."""
    return get_scalar_from_tensor(
        self._get_base_optimizer()._decayed_lr(var_dtype=tf.float32))  # pylint:disable=protected-access

  def _get_base_optimizer(self) -> tf.keras.optimizers.Optimizer:
    """Get the base optimizer used by the current model."""

    optimizer = self.model.optimizer

    # The optimizer might be wrapped by another class, so unwrap it
    while hasattr(optimizer, '_optimizer'):
      optimizer = optimizer._optimizer  # pylint:disable=protected-access

    return optimizer

class MovingAverageCallback(tf.keras.callbacks.Callback):
  """A Callback to be used with a `MovingAverage` optimizer.

  Applies moving average weights to the model during validation time to test
  and predict on the averaged weights rather than the current model weights.
  Once training is complete, the model weights will be overwritten with the
  averaged weights (by default).

  Attributes:
    overwrite_weights_on_train_end: Whether to overwrite the current model
      weights with the averaged weights from the moving average optimizer.
    **kwargs: Any additional callback arguments.
  """

  def __init__(self,
               overwrite_weights_on_train_end: bool = False,
               **kwargs):
    super(MovingAverageCallback, self).__init__(**kwargs)
    self.overwrite_weights_on_train_end = overwrite_weights_on_train_end

  def set_model(self, model: tf.keras.Model):
    super(MovingAverageCallback, self).set_model(model)
    assert isinstance(self.model.optimizer,
                      optimizer_factory.MovingAverage)
    self.model.optimizer.shadow_copy(self.model)

  def on_test_begin(self, logs: MutableMapping[Text, Any] = None):
    self.model.optimizer.swap_weights()

  def on_test_end(self, logs: MutableMapping[Text, Any] = None):
    self.model.optimizer.swap_weights()

  def on_train_end(self, logs: MutableMapping[Text, Any] = None):
    if self.overwrite_weights_on_train_end:
      self.model.optimizer.assign_average_vars(self.model.variables)


class AverageModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  """Saves and, optionally, assigns the averaged weights.

  Taken from tfa.callbacks.AverageModelCheckpoint.

  Attributes:
    update_weights: If True, assign the moving average weights
      to the model, and save them. If False, keep the old
      non-averaged weights, but the saved model uses the
      average weights.
    See `tf.keras.callbacks.ModelCheckpoint` for the other args.
  """

  def __init__(
      self,
      update_weights: bool,
      filepath: str,
      monitor: str = 'val_loss',
      verbose: int = 0,
      save_best_only: bool = False,
      save_weights_only: bool = False,
      mode: str = 'auto',
      save_freq: str = 'epoch',
      **kwargs):
    self.update_weights = update_weights
    super().__init__(
        filepath,
        monitor,
        verbose,
        save_best_only,
        save_weights_only,
        mode,
        save_freq,
        **kwargs)

  def set_model(self, model):
    if not isinstance(model.optimizer, optimizer_factory.MovingAverage):
      raise TypeError(
          'AverageModelCheckpoint is only used when training'
          'with MovingAverage')
    return super().set_model(model)

  def _save_model(self, epoch, logs):
    assert isinstance(self.model.optimizer, optimizer_factory.MovingAverage)

    if self.update_weights:
      self.model.optimizer.assign_average_vars(self.model.variables)
      return super()._save_model(epoch, logs)
    else:
      # Note: `model.get_weights()` gives us the weights (non-ref)
      # whereas `model.variables` returns references to the variables.
      non_avg_weights = self.model.get_weights()
      self.model.optimizer.assign_average_vars(self.model.variables)
      # result is currently None, since `super._save_model` doesn't
      # return anything, but this may change in the future.
      result = super()._save_model(epoch, logs)
      self.model.set_weights(non_avg_weights)
      return result



class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp

  def __repr__(self):
    return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
        self.batch_index, self.timestamp)



class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps, logger, logdir=None):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
      logdir: Optional directory to write TensorBoard summaries.
    """
    # TODO(wcromar): remove this parameter and rely on `logs` parameter of
    # on_train_batch_end()
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps
    self.last_log_step = 0
    self.steps_before_epoch = 0
    self.steps_in_epoch = 0
    self.start_time = None
    self.logger = logger
    self.step_per_epoch = 0

    if logdir:
      self.summary_writer = tf.summary.create_file_writer(logdir)
    else:
      self.summary_writer = None

    # Logs start of step 1 then end of each step based on log_steps interval.
    self.timestamp_log = []

    # Records the time each epoch takes to run from start to finish of epoch.
    self.epoch_runtime_log = []
    self.throughput = []

  @property
  def global_steps(self):
    """The current 1-indexed global step."""
    return self.steps_before_epoch + self.steps_in_epoch

  @property
  def average_steps_per_second(self):
    """The average training steps per second across all epochs."""
    return (self.global_steps - self.step_per_epoch) / sum(self.epoch_runtime_log[1:])

  @property
  def average_examples_per_second(self):
    """The average number of training examples per second across all epochs."""
    # return self.average_steps_per_second * self.batch_size
    ind = int(0.1*len(self.throughput))
    return sum(self.throughput[ind:])/(len(self.throughput[ind:])+1)

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

    if self.summary_writer:
      self.summary_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start = time.time()

  def on_batch_begin(self, batch, logs=None):
    if not self.start_time:
      self.start_time = time.time()

    # Record the timestamp of the first global step
    if not self.timestamp_log:
      self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                               self.start_time))

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    self.steps_in_epoch = batch + 1
    steps_since_last_log = self.global_steps - self.last_log_step
    if steps_since_last_log >= self.log_steps:
      now = time.time()
      elapsed_time = now - self.start_time
      steps_per_second = steps_since_last_log / elapsed_time
      examples_per_second = steps_per_second * self.batch_size

      self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
      elapsed_time_str='{:.2f} seconds'.format(elapsed_time)
      self.logger.log(step='PARAMETER', data={'TimeHistory': elapsed_time_str, 'examples/second': examples_per_second, 'steps': (self.last_log_step, self.global_steps)})

      if self.summary_writer:
        with self.summary_writer.as_default():
          tf.summary.scalar('global_step/sec', steps_per_second,
                            self.global_steps)
          tf.summary.scalar('examples/sec', examples_per_second,
                            self.global_steps)

      self.last_log_step = self.global_steps
      self.start_time = None
      self.throughput.append(examples_per_second)

  def on_epoch_end(self, epoch, logs=None):
    if epoch == 0:
      self.step_per_epoch = self.steps_in_epoch
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)

    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0


class EvalTimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, logger, logdir=None):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
      logdir: Optional directory to write TensorBoard summaries.
    """
    # TODO(wcromar): remove this parameter and rely on `logs` parameter of
    # on_train_batch_end()
    self.batch_size = batch_size
    self.global_steps = 0
    self.batch_time = []
    self.eval_time = 0
    super(EvalTimeHistory, self).__init__()
    self.logger = logger


  @property
  def average_steps_per_second(self):
    """The average training steps per second across all epochs."""
    return (self.global_steps - 1) / self.eval_time

  @property
  def average_examples_per_second(self):
    """The average number of training examples per second across all epochs."""
    return self.average_steps_per_second * self.batch_size

  def on_test_batch_end(self, batch, logs=None):
    self.global_steps += 1
    self.batch_time.append(time.time() - self.test_begin)

  def on_test_batch_begin(self, epoch, logs=None):
    self.test_begin = time.time()

  def on_test_end(self, epoch, logs=None):
    self.eval_time = sum(self.batch_time) - self.batch_time[0]