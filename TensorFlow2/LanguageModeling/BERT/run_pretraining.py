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
"""Run masked LM/next sentence masked_lm pre-training for BERT in tf2.0."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import horovod.tensorflow as hvd
import os

# Import BERT model libraries.
from official.nlp import bert_models
import common_flags
import input_pipeline
import model_saving_utils
from official.modeling import model_training_utils
from official.nlp import bert_modeling as modeling
import optimization
import gpu_affinity
import dllogger_class
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import tpu_lib

flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')
# Model training specific flags.
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('num_steps_per_epoch', 1000,
                     'Total number of training steps to run per epoch.')
flags.DEFINE_float('warmup_steps', 10000,
                   'Warmup steps for Adam weight decay optimizer.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def get_pretrain_dataset_fn(input_file_pattern, seq_length,
                            max_predictions_per_seq, global_batch_size):
  """Returns input dataset from input file string."""
  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    input_patterns = input_file_pattern.split(',')
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    train_dataset = input_pipeline.create_pretrain_dataset(
        input_patterns,
        seq_length,
        max_predictions_per_seq,
        batch_size,
        is_training=True,
        input_pipeline_context=ctx,
        use_horovod=FLAGS.use_horovod)
    return train_dataset

  return _dataset_fn


def get_loss_fn(loss_factor=1.0):
  """Returns loss function for BERT pretraining."""

  def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
    return tf.keras.backend.mean(losses) * loss_factor

  return _bert_pretrain_loss_fn


def run_customized_training(strategy,
                            bert_config,
                            max_seq_length,
                            max_predictions_per_seq,
                            model_dir,
                            steps_per_epoch,
                            steps_per_loop,
                            epochs,
                            initial_lr,
                            warmup_steps,
                            input_files,
                            train_batch_size):
  """Run BERT pretrain model training using low-level API."""

  train_input_fn = get_pretrain_dataset_fn(input_files, max_seq_length,
                                           max_predictions_per_seq,
                                           train_batch_size)

  def _get_pretrain_model():
    """Gets a pretraining model."""
    pretrain_model, core_model = bert_models.pretrain_model(
        bert_config, max_seq_length, max_predictions_per_seq, float_type=tf.float16 if FLAGS.use_fp16 else tf.float32)
    pretrain_model.optimizer = optimization.create_optimizer(
        initial_lr, steps_per_epoch * epochs, warmup_steps, FLAGS.optimizer_type)
    if FLAGS.use_fp16:
      pretrain_model.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(pretrain_model.optimizer,
        dynamic=True)
    return pretrain_model, core_model

  dllogging = dllogger_class.dllogger_class(FLAGS.dllog_path)
  params = {'dllogging' : dllogging, 'FLAGS' : FLAGS}
  logging.info("init_lr = %f", initial_lr)
  trained_model = model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_pretrain_model,
      loss_fn=get_loss_fn(
          loss_factor=1.0 /
          strategy.num_replicas_in_sync if FLAGS.scale_loss and strategy else 1.0),
      model_dir=model_dir,
      train_input_fn=train_input_fn,
      steps_per_epoch=steps_per_epoch,
      num_accumulative_step=FLAGS.num_accumulation_steps,
      steps_per_loop=steps_per_loop,
      epochs=epochs,
      sub_model_export_name='pretrained/bert_model',
      init_checkpoint=FLAGS.init_checkpoint,
      hvd=hvd if FLAGS.use_horovod else None,
      params=params)

  return trained_model


def run_bert_pretrain(strategy):
  """Runs BERT pre-training."""

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  # Padding for divisibility by 8
  # if bert_config.vocab_size % 8 != 0:
  #   bert_config.vocab_size += 8 - bert_config.vocab_size % 8
  if strategy:
    logging.info('Training using customized training loop TF 2.0 with distrubuted'
                'strategy.')

  keras_utils.set_config_v2(FLAGS.enable_xla)
  # Runs customized training loop.
  return run_customized_training(
      strategy,
      bert_config,
      FLAGS.max_seq_length,
      FLAGS.max_predictions_per_seq,
      FLAGS.model_dir,
      FLAGS.num_steps_per_epoch,
      FLAGS.steps_per_loop,
      FLAGS.num_train_epochs,
      FLAGS.learning_rate * hvd.size() if FLAGS.use_horovod else FLAGS.learning_rate,
      FLAGS.warmup_steps,
      FLAGS.input_files,
      FLAGS.train_batch_size)


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
  if strategy:
    print('***** Number of cores used : ', strategy.num_replicas_in_sync)

  if FLAGS.use_horovod:
    if strategy:
      raise ValueError('Should not run horovod with distribution strategy')

    hvd.init()
    if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
      gpu_affinity.set_affinity(hvd.local_rank())

  if FLAGS.use_fp16:
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)

  run_bert_pretrain(strategy)


if __name__ == '__main__':
  app.run(main)
