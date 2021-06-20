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
"""BERT classification finetuning runner in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import horovod.tensorflow as hvd

# Import BERT model libraries.
from official.nlp import bert_models
import common_flags
import input_pipeline
import model_saving_utils
from official.modeling import model_training_utils
from official.nlp import bert_modeling as modeling
import optimization
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils

flags.DEFINE_enum(
    'mode', 'train_and_eval', ['train_and_eval', 'export_only'],
    'One of {"train_and_eval", "export_only"}. `train_and_eval`: '
    'trains the model and evaluates in the meantime. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`.')
flags.DEFINE_string('train_data_path', None,
                    'Path to training data for BERT classifier.')
flags.DEFINE_string('eval_data_path', None,
                    'Path to evaluation data for BERT classifier.')
# Model training specific flags.
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def get_loss_fn(num_classes, loss_factor=1.0):
  """Gets the classification loss function."""

  def classification_loss_fn(labels, logits):
    """Classification loss."""
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    loss *= loss_factor
    return loss

  return classification_loss_fn


def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size,
                   is_training, use_horovod):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_classifier_dataset(
        input_file_pattern,
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx,
        use_horovod=use_horovod)
    return dataset

  return _dataset_fn


def run_bert_classifier(strategy,
                        bert_config,
                        input_meta_data,
                        model_dir,
                        epochs,
                        steps_per_epoch,
                        steps_per_loop,
                        eval_steps,
                        warmup_steps,
                        initial_lr,
                        init_checkpoint,
                        train_input_fn,
                        eval_input_fn,
                        custom_callbacks=None,
                        run_eagerly=False,
                        use_keras_compile_fit=False):
  """Run BERT classifier training using low-level API."""
  max_seq_length = input_meta_data['max_seq_length']
  num_classes = input_meta_data['num_labels']

  def _get_classifier_model():
    """Gets a classifier model."""
    classifier_model, core_model = (
        bert_models.classifier_model(
            bert_config,
            tf.float32,
            num_classes,
            max_seq_length,
            hub_module_url=FLAGS.hub_module_url))
    classifier_model.optimizer = optimization.create_optimizer(
        initial_lr, steps_per_epoch * epochs, warmup_steps)
    if FLAGS.fp16_implementation == 'graph_rewrite':
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      classifier_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          classifier_model.optimizer)
    return classifier_model, core_model

  # During distributed training, loss used for gradient computation is
  # summed over from all replicas. When Keras compile/fit() API is used,
  # the fit() API internally normalizes the loss by dividing the loss by
  # the number of replicas used for computation. However, when custom
  # training loop is used this is not done automatically and should be
  # done manually by the end user.
  loss_multiplier = 1.0
  if FLAGS.scale_loss and strategy and not use_keras_compile_fit:
    loss_multiplier = 1.0 / strategy.num_replicas_in_sync

  loss_fn = get_loss_fn(num_classes, loss_factor=loss_multiplier)

  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
  def metric_fn():
    return tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

  if use_keras_compile_fit:
    # Start training using Keras compile/fit API.
    logging.info('Training using TF 2.0 Keras compile/fit API with '
                 'distribution strategy.')
    return run_keras_compile_fit(
        model_dir,
        strategy,
        _get_classifier_model,
        train_input_fn,
        eval_input_fn,
        loss_fn,
        metric_fn,
        init_checkpoint,
        epochs,
        steps_per_epoch,
        eval_steps,
        custom_callbacks=None)

  # Use user-defined loop to start training.
  logging.info('Training using customized training loop TF 2.0 with '
               'distribution strategy.')
  return model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_classifier_model,
      loss_fn=loss_fn,
      model_dir=model_dir,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=steps_per_loop,
      epochs=epochs,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      eval_steps=eval_steps,
      init_checkpoint=init_checkpoint,
      metric_fn=metric_fn,
      hvd=hvd if FLAGS.use_horovod else None,
      custom_callbacks=custom_callbacks,
      run_eagerly=run_eagerly)


def run_keras_compile_fit(model_dir,
                          strategy,
                          model_fn,
                          train_input_fn,
                          eval_input_fn,
                          loss_fn,
                          metric_fn,
                          init_checkpoint,
                          epochs,
                          steps_per_epoch,
                          eval_steps,
                          custom_callbacks=None):
  """Runs BERT classifier model using Keras compile/fit API."""

  with distribution_utils.get_strategy_scope(strategy):
    training_dataset = train_input_fn()
    evaluation_dataset = eval_input_fn()
    bert_model, sub_model = model_fn()
    optimizer = bert_model.optimizer

    if init_checkpoint:
      checkpoint = tf.train.Checkpoint(model=sub_model)
      checkpoint.restore(init_checkpoint).assert_existing_objects_matched()

    bert_model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn()])

    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint_path = os.path.join(model_dir, 'checkpoint')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True)

    if custom_callbacks is not None:
      custom_callbacks += [summary_callback, checkpoint_callback]
    else:
      custom_callbacks = [summary_callback, checkpoint_callback]

    bert_model.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=eval_steps,
        callbacks=custom_callbacks)

    return bert_model


def export_classifier(model_export_path, input_meta_data,
                      restore_model_using_load_weights,
                      bert_config, model_dir):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.
    restore_model_using_load_weights: Whether to use checkpoint.restore() API
      for custom checkpoint or to use model.load_weights() API.
      There are 2 different ways to save checkpoints. One is using
      tf.train.Checkpoint and another is using Keras model.save_weights().
      Custom training loop implementation uses tf.train.Checkpoint API
      and Keras ModelCheckpoint callback internally uses model.save_weights()
      API. Since these two API's cannot be used together, model loading logic
      must be take into account how model checkpoint was saved.
    bert_config: Bert configuration file to define core bert layers.
    model_dir: The directory where the model weights and training/evaluation
      summaries are stored.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  if not model_export_path:
    raise ValueError('Export path is not specified: %s' % model_export_path)
  if not model_dir:
    raise ValueError('Export path is not specified: %s' % model_dir)

  classifier_model = bert_models.classifier_model(
      bert_config, tf.float32, input_meta_data['num_labels'],
      input_meta_data['max_seq_length'])[0]

  model_saving_utils.export_bert_model(
      model_export_path,
      model=classifier_model,
      checkpoint_dir=model_dir,
      restore_model_using_load_weights=restore_model_using_load_weights)


def run_bert(strategy,
             input_meta_data,
             train_input_fn=None,
             eval_input_fn=None):
  """Run BERT training."""
  if FLAGS.model_type == 'bert':
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  else:
    assert FLAGS.model_type == 'albert'
    bert_config = modeling.AlbertConfig.from_json_file(FLAGS.bert_config_file)
  if FLAGS.mode == 'export_only':
    # As Keras ModelCheckpoint callback used with Keras compile/fit() API
    # internally uses model.save_weights() to save checkpoints, we must
    # use model.load_weights() when Keras compile/fit() is used.
    export_classifier(FLAGS.model_export_path, input_meta_data,
                      FLAGS.use_keras_compile_fit,
                      bert_config, FLAGS.model_dir)
    return

  if FLAGS.mode != 'train_and_eval':
    raise ValueError('Unsupported mode is specified: %s' % FLAGS.mode)
  # Enables XLA in Session Config. Should not be set for TPU.
  keras_utils.set_config_v2(FLAGS.enable_xla)

  epochs = FLAGS.num_train_epochs
  train_data_size = input_meta_data['train_data_size']
  global_batch_size = FLAGS.train_batch_size
  learning_rate = FLAGS.learning_rate
  if FLAGS.use_horovod:
    global_batch_size *= hvd.size()
    learning_rate *= hvd.size()
  steps_per_epoch = int(train_data_size / global_batch_size)
  warmup_steps = int(epochs * train_data_size * 0.1 / global_batch_size)
  eval_steps = int(
      math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))

  if strategy:
    # Runs customized training loop.
    logging.info('Training using customized training loop TF 2.0 with distrubuted'
                'strategy.')
  trained_model = run_bert_classifier(
      strategy,
      bert_config,
      input_meta_data,
      FLAGS.model_dir,
      epochs,
      steps_per_epoch,
      FLAGS.steps_per_loop,
      eval_steps,
      warmup_steps,
      learning_rate,
      FLAGS.init_checkpoint,
      train_input_fn,
      eval_input_fn,
      run_eagerly=FLAGS.run_eagerly,
      use_keras_compile_fit=FLAGS.use_keras_compile_fit)

  if FLAGS.model_export_path:
    model_saving_utils.export_bert_model(
        FLAGS.model_export_path,
        model=trained_model,
        restore_model_using_load_weights=FLAGS.use_keras_compile_fit)
  return trained_model


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  if FLAGS.use_fp16:
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)

  if FLAGS.use_horovod:
    if strategy:
      raise ValueError('Should not run horovod with distribution strategy')

    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'

  max_seq_length = input_meta_data['max_seq_length']
  train_input_fn = get_dataset_fn(
      FLAGS.train_data_path,
      max_seq_length,
      FLAGS.train_batch_size,
      is_training=True,
      use_horovod=FLAGS.use_horovod)
  eval_input_fn = get_dataset_fn(
      FLAGS.eval_data_path,
      max_seq_length,
      FLAGS.eval_batch_size,
      is_training=False,
      use_horovod=FLAGS.use_horovod)

  run_bert(strategy, input_meta_data, train_input_fn, eval_input_fn)

if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
