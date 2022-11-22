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
"""The main training script."""
import os
import time
from mpi4py import MPI
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import dllogger as DLLogger

from model import anchors, callback_builder, coco_metric, dataloader
from model import efficientdet_keras, label_util, optimizer_builder, postprocess
from utils import hparams_config, model_utils, setup, train_lib, util_keras
from utils.horovod_utils import is_main_process, get_world_size, get_rank

# Model specific paramenters
flags.DEFINE_string('training_mode', 'traineval', '(train/train300/traineval)')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name.')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_integer('batch_size', 64, 'training local batch size')
flags.DEFINE_integer('eval_batch_size', 64, 'evaluation local batch size')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch (coco default is 117266)')
flags.DEFINE_integer('num_epochs', None, 'Number of epochs for training')
flags.DEFINE_bool('benchmark', False, 'Train for a fixed number of steps for performance')
flags.DEFINE_integer('benchmark_steps', 100, 'Train for these many steps to benchmark training performance')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')
flags.DEFINE_bool('use_xla', True, 'Use XLA')
flags.DEFINE_bool('amp', True, 'Enable mixed precision training')
flags.DEFINE_bool('set_num_threads', True, 'Set inter-op and intra-op parallelism threads')
flags.DEFINE_string('log_filename', 'time_log.txt', 'Filename for dllogger logs')
flags.DEFINE_integer('log_steps', 1, 'Interval of steps between logging of batch level stats')
flags.DEFINE_bool('lr_tb', False, 'Log learning rate at each step to TB')
flags.DEFINE_bool('enable_map_parallelization', True, 'Parallelize stateless map transformations in dataloader')
flags.DEFINE_integer('checkpoint_period', 10, 'Save ema model weights after every X epochs for eval')
flags.DEFINE_string('pretrained_ckpt', None,
                    'Start training from this EfficientDet checkpoint.')
flags.DEFINE_string('backbone_init', None,
                    'Initialize backbone weights from checkpoint in this directory.')
flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')
flags.DEFINE_float('lr', None, 'Learning rate')
flags.DEFINE_float('warmup_value', 0.0001, 'Initial warmup value')
flags.DEFINE_float('warmup_epochs', None, 'Number of warmup epochs')
flags.DEFINE_integer('seed', None, 'Random seed')    
flags.DEFINE_bool('debug', False, 'Enable debug mode')
flags.DEFINE_bool('time_history', True, 'Get time history')
flags.DEFINE_bool('validate', False, 'Get validation loss after each epoch')
flags.DEFINE_string('val_file_pattern', None,
                    'Glob for eval tfrecords, e.g. coco/val-*.tfrecord.')
flags.DEFINE_string(
    'val_json_file', None,
    'COCO validation JSON containing golden bounding boxes. If None, use the '
    'ground truth from the dataloader. Ignored if testdev_dir is not None.')
flags.DEFINE_string('testdev_dir', None,
                    'COCO testdev dir. If not None, ignorer val_json_file.')
flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                     'evaluation.')
FLAGS = flags.FLAGS


def main(_):

  # get e2e training time
  begin = time.time()
  logging.info("Training started at: {}".format(time.asctime()))

  hvd.init()

  # Parse and override hparams
  config = hparams_config.get_detection_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  if FLAGS.num_epochs:  # NOTE: remove this flag after updating all docs.
    config.num_epochs = FLAGS.num_epochs
  if FLAGS.lr:
    config.learning_rate = FLAGS.lr
  if FLAGS.warmup_value:
    config.lr_warmup_init = FLAGS.warmup_value
  if FLAGS.warmup_epochs:
    config.lr_warmup_epoch = FLAGS.warmup_epochs
  config.backbone_init = FLAGS.backbone_init
  config.mixed_precision = FLAGS.amp
  config.image_size = model_utils.parse_image_size(config.image_size)

  # get eval config
  eval_config = hparams_config.get_detection_config(FLAGS.model_name)
  eval_config.override(FLAGS.hparams)
  eval_config.val_json_file = FLAGS.val_json_file
  eval_config.val_file_pattern = FLAGS.val_file_pattern
  eval_config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
  eval_config.drop_remainder = False  # eval all examples w/o drop.
  eval_config.image_size = model_utils.parse_image_size(eval_config['image_size'])

  # setup
  setup.set_flags(FLAGS, config, training=True)

  if FLAGS.debug:
    tf.config.experimental_run_functions_eagerly(True)
    tf.debugging.set_log_device_placement(True)
    tf.random.set_seed(111111)
    logging.set_verbosity(logging.DEBUG)

  # Check data path
  if FLAGS.training_file_pattern is None or FLAGS.val_file_pattern is None or FLAGS.val_json_file is None:
    raise RuntimeError('You must specify --training_file_pattern, --val_file_pattern and --val_json_file  for training.')

  steps_per_epoch = (FLAGS.num_examples_per_epoch + (FLAGS.batch_size * get_world_size()) - 1) // (FLAGS.batch_size * get_world_size())
  if FLAGS.benchmark == True:
    # For ci perf training runs, run for a fixed number of iterations per epoch
    steps_per_epoch = FLAGS.benchmark_steps
  params = dict(
      config.as_dict(),
      model_name=FLAGS.model_name,
      model_dir=FLAGS.model_dir,
      steps_per_epoch=steps_per_epoch,
      checkpoint_period=FLAGS.checkpoint_period,
      batch_size=FLAGS.batch_size,
      num_shards=get_world_size(),
      val_json_file=FLAGS.val_json_file,
      testdev_dir=FLAGS.testdev_dir,
      mode='train')
  logging.info('Training params: {}'.format(params))

  # make output dir if it does not exist
  tf.io.gfile.makedirs(FLAGS.model_dir)

  # dllogger setup
  backends = []
  if is_main_process():
    log_path = os.path.join(FLAGS.model_dir, FLAGS.log_filename)
    backends+=[
      JSONStreamBackend(verbosity=Verbosity.VERBOSE, filename=log_path),
      StdOutBackend(verbosity=Verbosity.DEFAULT)]
    
  DLLogger.init(backends=backends)
  DLLogger.metadata('avg_fps_training', {'unit': 'images/s'})
  DLLogger.metadata('avg_fps_training_per_GPU', {'unit': 'images/s'})
  DLLogger.metadata('avg_latency_training', {'unit': 's'})
  DLLogger.metadata('training_loss', {'unit': None})
  DLLogger.metadata('e2e_training_time', {'unit': 's'})

  def get_dataset(is_training, params):
    file_pattern = (
        FLAGS.training_file_pattern
        if is_training else FLAGS.val_file_pattern)
    if not file_pattern:
      raise ValueError('No matching files.')

    return dataloader.InputReader(
        file_pattern,
        is_training=is_training,
        use_fake_data=FLAGS.use_fake_data,
        max_instances_per_image=config.max_instances_per_image,
        enable_map_parallelization=FLAGS.enable_map_parallelization)(
            params)

  num_samples = (FLAGS.eval_samples + get_world_size() - 1) // get_world_size()
  num_samples = (num_samples + FLAGS.eval_batch_size - 1) // FLAGS.eval_batch_size
  eval_config.num_samples = num_samples
  
  def get_eval_dataset(eval_config):
    dataset = dataloader.InputReader(
      FLAGS.val_file_pattern,
      is_training=False,
      max_instances_per_image=eval_config.max_instances_per_image)(
          eval_config, batch_size=FLAGS.eval_batch_size)
    dataset = dataset.shard(get_world_size(), get_rank())
    dataset = dataset.take(num_samples)
    return dataset

  eval_dataset = get_eval_dataset(eval_config)

  # pick focal loss implementation
  focal_loss = train_lib.StableFocalLoss(
                params['alpha'],
                params['gamma'],
                label_smoothing=params['label_smoothing'],
                reduction=tf.keras.losses.Reduction.NONE)

  model = train_lib.EfficientDetNetTrain(params['model_name'], config)
  model.build((None, *config.image_size, 3))
  model.compile(
      optimizer=optimizer_builder.get_optimizer(params),
      loss={
          'box_loss':
              train_lib.BoxLoss(
                  params['delta'], reduction=tf.keras.losses.Reduction.NONE),
          'box_iou_loss':
              train_lib.BoxIouLoss(
                  params['iou_loss_type'],
                  params['min_level'],
                  params['max_level'],
                  params['num_scales'],
                  params['aspect_ratios'],
                  params['anchor_scale'],
                  params['image_size'],
                  reduction=tf.keras.losses.Reduction.NONE),
          'class_loss': focal_loss,
          'seg_loss':
              tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True,
                  reduction=tf.keras.losses.Reduction.NONE)
      })
  train_from_epoch = util_keras.restore_ckpt(model, params['model_dir'], 
                      config.moving_average_decay, steps_per_epoch=steps_per_epoch)

  print("training_mode: {}".format(FLAGS.training_mode))
  callbacks = callback_builder.get_callbacks(params, FLAGS.training_mode, eval_config, eval_dataset, 
                DLLogger, FLAGS.time_history, FLAGS.log_steps, FLAGS.lr_tb, FLAGS.benchmark)

  history = model.fit(
      get_dataset(True, params=params),
      epochs=params['num_epochs'],
      steps_per_epoch=steps_per_epoch,
      initial_epoch=train_from_epoch,
      callbacks=callbacks,
      verbose=1 if is_main_process() else 0,
      validation_data=get_dataset(False, params=params) if FLAGS.validate else None,
      validation_steps=(FLAGS.eval_samples // FLAGS.eval_batch_size) if FLAGS.validate else None)

  if is_main_process():
    model.save_weights(os.path.join(FLAGS.model_dir, 'ckpt-final'))

  # log final stats
  stats = {}
  for callback in callbacks:
    if isinstance(callback, callback_builder.TimeHistory):
      if callback.epoch_runtime_log:
        stats['avg_fps_training'] = callback.average_examples_per_second
        stats['avg_fps_training_per_GPU'] = callback.average_examples_per_second / get_world_size()
        stats['avg_latency_training'] = callback.average_time_per_iteration

  if history and history.history:
    train_hist = history.history
    #Gets final loss from training.
    stats['training_loss'] = float(hvd.allreduce(tf.constant(train_hist['loss'][-1], dtype=tf.float32), average=True))

  if os.path.exists(os.path.join(FLAGS.model_dir,'ema_weights')):    
    ckpt_epoch = "%02d" % sorted(set([int(f.rsplit('.')[0].rsplit('-')[1])
                        for f in os.listdir(os.path.join(FLAGS.model_dir,'ema_weights'))
                        if 'emackpt' in f]), reverse=True)[0]
    ckpt = os.path.join(FLAGS.model_dir, 'ema_weights', 'emackpt-' + str(ckpt_epoch))
    util_keras.restore_ckpt(model, ckpt, eval_config.moving_average_decay,
                            steps_per_epoch=0, skip_mismatch=False, expect_partial=True)
    if is_main_process():
      model.save(os.path.join(FLAGS.model_dir, 'emackpt-final'))
  else:
    ckpt_epoch = 'final'
    ckpt = os.path.join(FLAGS.model_dir, 'ckpt-' + ckpt_epoch)
    if is_main_process():
      model.save(os.path.join(FLAGS.model_dir, 'ckpt-' + ckpt_epoch))

  # Start evaluation of final ema checkpoint
  logging.set_verbosity(logging.WARNING)

  @tf.function
  def model_fn(images, labels):
    cls_outputs, box_outputs = model(images, training=False)
    detections = postprocess.generate_detections(eval_config, cls_outputs, box_outputs,
                                            labels['image_scales'],
                                            labels['source_ids'])

    tf.numpy_function(evaluator.update_state,
                      [labels['groundtruth_data'], 
                      postprocess.transform_detections(detections)], [])

  if FLAGS.benchmark == False and (FLAGS.training_mode == 'train' or FLAGS.num_epochs < 200):

    # Evaluator for AP calculation.
    label_map = label_util.get_label_map(eval_config.label_map)
    evaluator = coco_metric.EvaluationMetric(
      filename=eval_config.val_json_file, label_map=label_map)

    evaluator.reset_states()

    # evaluate all images.
    pbar = tf.keras.utils.Progbar(num_samples)
    for i, (images, labels) in enumerate(eval_dataset):
      model_fn(images, labels)
      if is_main_process():
        pbar.update(i)

    # gather detections from all ranks
    evaluator.gather()

    if is_main_process():
      # compute the final eval results.
      metrics = evaluator.result()
      metric_dict = {}
      for i, name in enumerate(evaluator.metric_names):
        metric_dict[name] = metrics[i]

      if label_map:
        for i, cid in enumerate(sorted(label_map.keys())):
          name = 'AP_/%s' % label_map[cid]
          metric_dict[name] = metrics[i + len(evaluator.metric_names)]

      # csv format
      csv_metrics = ['AP','AP50','AP75','APs','APm','APl']
      csv_format = ",".join([str(ckpt_epoch)] + [str(round(metric_dict[key] * 100, 2)) for key in csv_metrics])
      print(FLAGS.model_name, metric_dict, "csv format:", csv_format)
      DLLogger.log(step=(), data={'epoch': ckpt_epoch,
                    'validation_accuracy_mAP': round(metric_dict['AP'] * 100, 2)})
      DLLogger.flush()

    MPI.COMM_WORLD.Barrier()

  if is_main_process():
    stats['e2e_training_time'] = time.time() - begin
    DLLogger.log(step=(), data=stats)
    DLLogger.flush()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
