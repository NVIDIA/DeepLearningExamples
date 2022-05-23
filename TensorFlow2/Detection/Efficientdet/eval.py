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
"""Eval libraries."""
import os
from mpi4py import MPI
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import horovod.tensorflow.keras as hvd

from model import anchors
from model import coco_metric
from model import dataloader
from model import efficientdet_keras
from model import label_util
from model import postprocess
from utils import hparams_config
from utils import model_utils
from utils import util_keras
from utils.horovod_utils import get_rank, get_world_size, is_main_process

flags.DEFINE_integer('eval_samples', 5000, 'Number of eval samples.')
flags.DEFINE_string('val_file_pattern', None,
                    'Glob for eval tfrecords, e.g. coco/val-*.tfrecord.')
flags.DEFINE_string('val_json_file', None,
                    'Groudtruth, e.g. annotations/instances_val2017.json.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name to use.')
flags.DEFINE_string('ckpt_path', None, 'Checkpoint path to evaluate')
flags.DEFINE_integer('batch_size', 8, 'Local batch size.')
flags.DEFINE_string('only_this_epoch', None, 'Evaluate only this epoch checkpoint.')       
flags.DEFINE_bool('enable_map_parallelization', True, 'Parallelize stateless map transformations in dataloader')
flags.DEFINE_bool('amp', True, 'Use mixed precision for eval.')           
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file.')
FLAGS = flags.FLAGS


def main(_):

  hvd.init()
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  if FLAGS.amp:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
  else:
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  config.val_json_file = FLAGS.val_json_file
  config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
  config.drop_remainder = False  # eval all examples w/o drop.
  config.image_size = model_utils.parse_image_size(config['image_size'])

  @tf.function
  def model_fn(images, labels):
    cls_outputs, box_outputs = model(images, training=False)
    detections = postprocess.generate_detections(config, cls_outputs, box_outputs,
                                            labels['image_scales'],
                                            labels['source_ids'])

    tf.numpy_function(evaluator.update_state,
                      [labels['groundtruth_data'], 
                      postprocess.transform_detections(detections)], [])

  # Network
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((None, *config.image_size, 3))

  # dataset
  batch_size = FLAGS.batch_size   # local batch size.
  ds = dataloader.InputReader(
      FLAGS.val_file_pattern,
      is_training=False,
      max_instances_per_image=config.max_instances_per_image,
      enable_map_parallelization=FLAGS.enable_map_parallelization)(
          config, batch_size=batch_size)
  ds = ds.shard(get_world_size(), get_rank())

  # Evaluator for AP calculation.
  label_map = label_util.get_label_map(config.label_map)
  evaluator = coco_metric.EvaluationMetric(
      filename=config.val_json_file, label_map=label_map)

  util_keras.restore_ckpt(model, FLAGS.ckpt_path, config.moving_average_decay,
                              steps_per_epoch=0, skip_mismatch=False, expect_partial=True)

  if FLAGS.eval_samples:
    num_samples = (FLAGS.eval_samples + get_world_size() - 1) // get_world_size()
    num_samples = (num_samples + batch_size - 1) // batch_size
    ds = ds.take(num_samples)
  evaluator.reset_states()

  # evaluate all images.
  pbar = tf.keras.utils.Progbar(num_samples)
  for i, (images, labels) in enumerate(ds):
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
    csv_format = ",".join([str(round(metric_dict[key] * 100, 2)) for key in csv_metrics])
    print(FLAGS.model_name, metric_dict, "csv format:", csv_format)
    
  MPI.COMM_WORLD.Barrier()


if __name__ == '__main__':
  flags.mark_flag_as_required('val_file_pattern')
  flags.mark_flag_as_required('val_json_file')
  flags.mark_flag_as_required('ckpt_path')
  logging.set_verbosity(logging.WARNING)
  app.run(main)
