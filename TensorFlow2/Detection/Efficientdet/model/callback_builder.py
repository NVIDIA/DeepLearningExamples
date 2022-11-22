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
"""Callback related utils."""
from concurrent import futures
import os
from mpi4py import MPI
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras.callbacks as hvd_callbacks
from tensorflow_addons.optimizers import MovingAverage
from typeguard import typechecked
from typing import Any, List, MutableMapping, Text

from model import inference, optimizer_builder
from utils import model_utils
from model import efficientdet_keras, coco_metric, label_util, postprocess
from utils.horovod_utils import get_world_size, is_main_process


class DisplayCallback(tf.keras.callbacks.Callback):
  """Display inference result callback."""

  def __init__(self, sample_image, output_dir, update_freq=1):
    super().__init__()
    image_file = tf.io.read_file(sample_image)
    self.sample_image = tf.expand_dims(
        tf.image.decode_jpeg(image_file, channels=3), axis=0)
    self.executor = futures.ThreadPoolExecutor(max_workers=1)
    self.update_freq = update_freq
    self.output_dir = output_dir

  def set_model(self, model: tf.keras.Model):
    self.train_model = model
    with tf.device('/cpu:0'):
      self.model = efficientdet_keras.EfficientDetModel(config=model.config)
    height, width = model_utils.parse_image_size(model.config.image_size)
    self.model.build((1, height, width, 3))
    self.file_writer = tf.summary.create_file_writer(self.output_dir)
    self.min_score_thresh = self.model.config.nms_configs['score_thresh'] or 0.4
    self.max_boxes_to_draw = (
        self.model.config.nms_configs['max_output_size'] or 100)

  def on_epoch_end(self, epoch, logs=None):
    if epoch % self.update_freq == 0:
      self.executor.submit(self.draw_inference, epoch)

  @tf.function
  def inference(self):
    return self.model(self.sample_image, training=False)

  def draw_inference(self, epoch):
    self.model.set_weights(self.train_model.get_weights())
    boxes, scores, classes, valid_len = self.inference()
    length = valid_len[0]
    image = inference.visualize_image(
        self.sample_image[0],
        boxes[0].numpy()[:length],
        classes[0].numpy().astype(np.int)[:length],
        scores[0].numpy()[:length],
        label_map=self.model.config.label_map,
        min_score_thresh=self.min_score_thresh,
        max_boxes_to_draw=self.max_boxes_to_draw)

    with self.file_writer.as_default():
      tf.summary.image('Test image', tf.expand_dims(image, axis=0), step=epoch)


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

  def __init__(self, batch_size, logger, log_steps=1, logdir=None):
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
    self.latency = []
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
    return sum(self.throughput[ind:])/(len(self.throughput[ind:]))

  @property
  def average_time_per_iteration(self):
    """The average time per iteration in seconds across all epochs."""
    ind = int(0.1*len(self.latency))
    return sum(self.latency[ind:])/(len(self.latency[ind:]))

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
      self.logger.log(step='PARAMETER', data={'Latency': elapsed_time_str, 'fps': examples_per_second, 'steps': (self.last_log_step, self.global_steps)})
      self.logger.flush()

      if self.summary_writer:
        with self.summary_writer.as_default():
          tf.summary.scalar('global_step/sec', steps_per_second,
                            self.global_steps)
          tf.summary.scalar('examples/sec', examples_per_second,
                            self.global_steps)

      self.last_log_step = self.global_steps
      self.start_time = None
      self.latency.append(elapsed_time)
      self.throughput.append(examples_per_second)

  def on_epoch_end(self, epoch, logs=None):
    if epoch == 0:
      self.step_per_epoch = self.steps_in_epoch
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)

    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0


class LRTensorBoard(tf.keras.callbacks.Callback):

  def __init__(self, log_dir, **kwargs):
    super().__init__(**kwargs)
    self.summary_writer = tf.summary.create_file_writer(log_dir)
    self.steps_before_epoch = 0
    self.steps_in_epoch = 0

  @property
  def global_steps(self):
    """The current 1-indexed global step."""
    return self.steps_before_epoch + self.steps_in_epoch

  def on_batch_end(self, batch, logs=None):
    self.steps_in_epoch = batch + 1

    lr = self.model.optimizer.lr(self.global_steps)
    with self.summary_writer.as_default():
      summary = tf.summary.scalar('learning_rate', lr, self.global_steps)

  def on_epoch_end(self, epoch, logs=None):
    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0

  def on_train_end(self, logs=None):
    self.summary_writer.flush()


class LoggingCallback(tf.keras.callbacks.Callback):
  def on_train_batch_end(self, batch, logs=None):
    print("Iter: {}".format(batch))
    for var in self.model.variables:
      # if 'dense' in var.name:
      #   continue
      print("Var: {} {}".format(var.name, var.value))
      try:
        slot = self.model.optimizer.get_slot(var, "average")
        print("Avg: {}".format(slot))
      except KeyError as e:
        print("{} does not have ema average slot".format(var.name))


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
    self.ema_opt = None

  def set_model(self, model: tf.keras.Model):
    super(MovingAverageCallback, self).set_model(model)
    self.ema_opt = fetch_optimizer(model, MovingAverage)
    self.ema_opt.shadow_copy(self.model.weights)

  def on_test_begin(self, logs: MutableMapping[Text, Any] = None):
    self.ema_opt.swap_weights()

  def on_test_end(self, logs: MutableMapping[Text, Any] = None):
    self.ema_opt.swap_weights()

  def on_train_end(self, logs: MutableMapping[Text, Any] = None):
    if self.overwrite_weights_on_train_end:
      self.ema_opt.assign_average_vars(self.model.variables)


class AverageModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  """Saves and, optionally, assigns the averaged weights.

  Taken from tfa.callbacks.AverageModelCheckpoint [original class].
  NOTE1: The original class has a type check decorator, which prevents passing non-string save_freq (fix: removed)
  NOTE2: The original class may not properly handle layered (nested) optimizer objects (fix: use fetch_optimizer)

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

    super().__init__(
        filepath,
        monitor,
        verbose,
        save_best_only,
        save_weights_only,
        mode,
        save_freq,
        **kwargs)
    self.update_weights = update_weights
    self.ema_opt = None

  def set_model(self, model):
    self.ema_opt = fetch_optimizer(model, MovingAverage)
    return  super().set_model(model)

  def _save_model(self, epoch, batch, logs):
    assert isinstance(self.ema_opt, MovingAverage)

    if self.update_weights:
      self.ema_opt.assign_average_vars(self.model.variables)
      return super()._save_model(epoch, batch, logs)
    else:
      # Note: `model.get_weights()` gives us the weights (non-ref)
      # whereas `model.variables` returns references to the variables.
      non_avg_weights = self.model.get_weights()
      self.ema_opt.assign_average_vars(self.model.variables)
      # result is currently None, since `super._save_model` doesn't
      # return anything, but this may change in the future.
      result = super()._save_model(epoch, batch, logs)
      self.model.set_weights(non_avg_weights)
      return result


class StopEarlyCallback(tf.keras.callbacks.Callback):
  def __init__(self, num_epochs, stop_75, **kwargs):
    super(StopEarlyCallback, self).__init__(**kwargs)
    self.num_epochs = num_epochs
    self.stop_75 = stop_75

  def on_epoch_end(self, epoch, logs=None):
    if ((epoch + 1) > (0.75 * self.num_epochs) and self.stop_75) or ((epoch + 1) == 300):
      self.model.stop_training = True


class COCOEvalCallback(tf.keras.callbacks.Callback):
  def __init__(self, eval_dataset, eval_freq, start_eval_epoch, eval_params, logger, **kwargs):
    super(COCOEvalCallback, self).__init__(**kwargs)
    self.dataset = eval_dataset
    self.eval_freq = eval_freq
    self.start_eval_epoch = start_eval_epoch
    self.eval_params = eval_params
    self.ema_opt = None
    self.logger = logger

    label_map = label_util.get_label_map(eval_params['label_map'])
    self.evaluator = coco_metric.EvaluationMetric(
      filename=eval_params['val_json_file'], label_map=label_map)

    self.pbar = tf.keras.utils.Progbar(eval_params['num_samples'])

  def set_model(self, model):
    self.ema_opt = fetch_optimizer(model, MovingAverage)
    return super().set_model(model)

  @tf.function
  def eval_model_fn(self, images, labels):
    cls_outputs, box_outputs = self.model(images, training=False)
    detections = postprocess.generate_detections(self.eval_params, cls_outputs, box_outputs,
                                            labels['image_scales'],
                                            labels['source_ids'])

    tf.numpy_function(self.evaluator.update_state,
                      [labels['groundtruth_data'], 
                      postprocess.transform_detections(detections)], [])

  def evaluate(self, epoch):
    if self.eval_params['moving_average_decay'] > 0:
      self.ema_opt.swap_weights() # get ema weights

    self.evaluator.reset_states()
    # evaluate all images.
    for i, (images, labels) in enumerate(self.dataset):
      self.eval_model_fn(images, labels)
      if is_main_process():
        self.pbar.update(i)

    # gather detections from all ranks
    self.evaluator.gather()

    # compute the final eval results.
    if is_main_process():
      metrics = self.evaluator.result()
      metric_dict = {}
      for i, name in enumerate(self.evaluator.metric_names):
        metric_dict[name] = metrics[i]

      # csv format
      csv_metrics = ['AP','AP50','AP75','APs','APm','APl']
      csv_format = ",".join([str(epoch+1)] + [str(round(metric_dict[key] * 100, 2)) for key in csv_metrics])
      print(metric_dict, "csv format:", csv_format)
      self.logger.log(step=(), data={'epoch': epoch+1,
                                  'validation_accuracy_mAP': round(metric_dict['AP'] * 100, 2)})

    if self.eval_params['moving_average_decay'] > 0:
      self.ema_opt.swap_weights() # get base weights
    
    MPI.COMM_WORLD.Barrier()

  def on_epoch_end(self, epoch, logs=None):
    if (epoch + 1) >= self.start_eval_epoch and (epoch + 1) % self.eval_freq == 0:
      self.evaluate(epoch)


def get_callbacks(
      params, training_mode, eval_params, eval_dataset, logger, 
      time_history=True, log_steps=1, lr_tb=True, benchmark=False
    ):
  """Get callbacks for given params."""
  callbacks = []
  if is_main_process():
    if benchmark == False:
      tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=params['model_dir'], profile_batch=0, histogram_freq = 1)
      callbacks.append(tb_callback)

    if params['moving_average_decay']:
      emackpt_callback = AverageModelCheckpoint(
        filepath=os.path.join(params['model_dir'], 'ema_weights', 'emackpt-{epoch:02d}'),
        update_weights=False,
        amp=params['mixed_precision'],
        verbose=1,
        save_freq='epoch',
        save_weights_only=True,
        period=params['checkpoint_period'])
      callbacks.append(emackpt_callback)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
      os.path.join(params['model_dir'], 'ckpt'),
      verbose=1,
      save_freq='epoch',
      save_weights_only=True,
      period=params['checkpoint_period'])
    callbacks.append(ckpt_callback)

    if time_history:
      time_callback = TimeHistory(params['batch_size'] * get_world_size(),
        logger=logger,
        logdir=params['model_dir'],
        log_steps=log_steps)
      callbacks.append(time_callback)

    # log LR in tensorboard
    if lr_tb == True and benchmark == False:
      callbacks.append(LRTensorBoard(log_dir=params['model_dir']))
  
  hvd_callback = hvd_callbacks.BroadcastGlobalVariablesCallback(0)
  callbacks.append(hvd_callback)

  # for large batch sizes training schedule of 350/400 epochs gives better mAP
  # but the best mAP is generally reached after 75% of the training schedule.
  # So we can stop training at that point or continue to train until 300 epochs
  stop_75 = False if 'eval' in training_mode or '300' in training_mode else True
  early_stopping = StopEarlyCallback(params['num_epochs'], stop_75=stop_75)
  callbacks.append(early_stopping)

  if 'eval' in training_mode:
    cocoeval = COCOEvalCallback(eval_dataset, 
                eval_freq=params['checkpoint_period'], 
                start_eval_epoch=200, 
                eval_params=eval_params,
                logger=logger)
    callbacks.append(cocoeval)

  if params['moving_average_decay']:
    callbacks.append(MovingAverageCallback())

  if params.get('sample_image', None):
    display_callback = DisplayCallback(
        params.get('sample_image', None),
        os.path.join(params['model_dir'], 'train'))
    callbacks.append(display_callback)

  return callbacks
