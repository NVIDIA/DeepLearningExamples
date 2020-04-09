# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import json
import os
import tensorflow as tf
import tensorflow_transform as tft

from trainer import features
from utils.hooks.benchmark_hooks import BenchmarkLoggingHook

import horovod.tensorflow as hvd

import dllogger

MODEL_TYPES = ['wide', 'deep', 'wide_n_deep']
WIDE, DEEP, WIDE_N_DEEP = MODEL_TYPES

# Default train dataset size
TRAIN_DATASET_SIZE = 59761827

def create_parser():
  """Initialize command line parser using arparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--model_type',
    help='Model type to train on',
    choices=MODEL_TYPES,
    default=WIDE_N_DEEP)
  parser.add_argument(
    '--canned_estimator',
    help='Use canned estimator instead of the experimental custom estimator',
    action='store_true',
    default=False)
  parser.add_argument(
    '--train_data_pattern',
    help='Pattern of training file names. For example if training files are train_000.tfrecord, \
    train_001.tfrecord then --train_data_pattern is train_*',
    type=str,
    default='/outbrain/tfrecords/train_*',
    nargs='+'
  )
  parser.add_argument(
    '--eval_data_pattern',
    help='Pattern of eval file names. For example if eval files are eval_000.tfrecord, \
    eval_001.tfrecord then --eval_data_pattern is eval_*',
    type=str,
    default='/outbrain/tfrecords/eval_*',
    nargs='+'
  )
  parser.add_argument(
    '--model_dir',
    help='Model Checkpoint will be saved here',
    type=str,
    default='/outbrain/checkpoints'
  )
  parser.add_argument(
    '--transformed_metadata_path',
    help='Path to transformed_metadata.', 
    type=str, 
    default='/outbrain/tfrecords'
  )
  parser.add_argument(
    '--deep_hidden_units',
    help='hidden units per layer, separated by spaces',
    default=[1024, 1024, 1024, 1024, 1024],
    type=int,
    nargs="+")
  parser.add_argument(
    '--prebatch_size',
    help='Size of the pre-batches in the tfrecords',
    default=4096,
    type=int)
  parser.add_argument(
    '--global_batch_size',
    help='Total training batch size',
    default=131072,
    type=int)
  parser.add_argument(
    '--eval_batch_size',
    help='Evaluation batch size',
    default=32768,
    type=int)
  parser.add_argument(
    '--eval_steps',
    help='Number of evaluation steps to perform',
    default=8,
    type=int)
  parser.add_argument(
    '--training_set_size',
    help='Number of samples in the training set',
    default=TRAIN_DATASET_SIZE,
    type=int)
  parser.add_argument(
    '--num_epochs',
     help='Number of epochs',
     default=120,
     type=int)
  parser.add_argument(
    '--save_checkpoints_secs',
    help='Minimal number of seconds between evaluations',
    default=600,
    type=int)
  parser.add_argument(
    '--save_checkpoints_steps',
    help='Training steps between saving checkpoints. If 0, then save_checkpoints_secs applies',
    default=0,
    type=int)
  
  parser.add_argument(
    '--xla',
    help='Enable XLA',
    default=False,
    action='store_true')
  parser.add_argument(
    '--gpu',
    help='Run computations on the GPU',
    default=False,
    action='store_true')
  parser.add_argument(
    '--amp',
    help='Attempt automatic mixed precision conversion',
    default=False,
    action='store_true')
  parser.add_argument(
    '--hvd',
    help='Use Horovod',
    action='store_true',
    default=False)
  
  # hyperparameters for linear part
  parser.add_argument(
    '--linear_l1_regularization',
    help='L1 regularization for linear model',
    type=float,
    default=0.0)
  parser.add_argument(
    '--linear_l2_regularization',
    help='L2 regularization for linear model',
    type=float,
    default=0.0)
  parser.add_argument(
    '--linear_learning_rate',
    help='Learning rate for linear model',
    type=float,
    default=0.2)

  # hyperparameters for deep part
  parser.add_argument(
    '--deep_l1_regularization',
    help='L1 regularization for deep model',
    type=float,
    default=0.0)
  parser.add_argument(
    '--deep_l2_regularization',
    help='L2 regularization for deep model',
    type=float,
    default=0.00)
  parser.add_argument(
    '--deep_learning_rate',
    help='Learning rate for deep model',
    type=float,
    default=1.0)
  parser.add_argument(
    '--deep_dropout',
    help='Dropout regularization for deep model',
    type=float,
    default=0.0)

  parser.add_argument(
    '--log_device_placement',
    help='Ask Tensorflow (via ConfigProto) to print device placement of nodes',
    default=False,
    action='store_true')

  parser.add_argument(
    '--predict',
    help='Only perform a prediction on the validation dataset, don\'t train',
    default=False,
    action='store_true')
  parser.add_argument(
    '--evaluate',
    help='Only perform an evaluation on the validation dataset, don\'t train',
    default=False,
    action='store_true')
  parser.add_argument(
    '--results_dir',
    type=str,
    help='Directory to store training results',
    default='/results')
  parser.add_argument(
    '--log_filename',
    type=str,
    help='Name of the file to store dlloger output',
    default='log.json')
  parser.add_argument(
    '--use_all_columns',
    help='Force using all features defined in the features.py file',
    default=False,
    action='store_true')
  parser.add_argument(
    '--shuffle_percentage',
    type=float,
    default=0.001,
    help='Size of the shuffle buffer from 0 to 1. 1 means that the shuffle buffer size will be equal to the size of the training dataset.')
  parser.add_argument(
    '--print_display_ids',
    help='Print the display ids processed by the input pipeline',
    default=False,
    action='store_true')
  parser.add_argument(
    '--eval_throttle_secs',
    help='Number of evaluation steps to perform.',
    default=600,
    type=int)
  parser.add_argument(
    '--reader_num_threads',
    default=12,
    type=int)
  parser.add_argument(
    '--parser_num_threads',
    default=3,
    type=int)
  parser.add_argument(
    '--prefetch_buffer_size',
    default=1,
    type=int)
  parser.add_argument(
    '--submission',
    action='store_true',
    default=False)
  parser.add_argument(
    '--benchmark',
    help='Collect performance metrics during training',
    action='store_true',
    default=False)
  parser.add_argument(
    '--benchmark_warmup_steps',
    help='Warmup before starg of benchmarking the training',
    type=int,
    default=50)
  parser.add_argument(
    '--benchmark_steps',
    help='Number of steps for train performance benchmark',
    type=int,
    default=100)

  return parser


def get_feature_columns(use_all_columns=False, force_subset=None):
  # adding the force_subset as a way to directly pass in column changes for testing/profiling
  assert not use_all_columns or force_subset is None, \
          'Cannot both use all columns and use only a subset; give only one argument'
  deep_columns, wide_columns = [], []

  if use_all_columns:
    training_columns = features.ALL_TRAINING_COLUMNS
  elif force_subset is not None:
    training_columns = force_subset
  else:
    training_columns = features.NV_TRAINING_COLUMNS

  tf.compat.v1.logging.warn('number of features: {}'.format(len(training_columns)))

  for column_name in training_columns:
    if column_name in features.HASH_BUCKET_SIZES:
      categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
          column_name,
          hash_bucket_size=features.HASH_BUCKET_SIZES[column_name],
          dtype=tf.int32)
      wide_columns.append(categorical_column)

    elif column_name in features.IDENTITY_NUM_BUCKETS:
      categorical_column = tf.feature_column.categorical_column_with_identity(
        column_name, num_buckets=features.IDENTITY_NUM_BUCKETS[column_name])
      wide_columns.append(categorical_column)

    else:
      columns = []
      if column_name in features.FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
        # add a categorical_column for column_name + "_binned"
        # just add the regular float column for now
        columns.append(tf.feature_column.numeric_column(
          column_name, shape=(1,)))
      elif column_name in features.FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
        # add a categorical_column for column_name + "_log_binned")
        columns.append(tf.feature_column.numeric_column(
          column_name + "_log_01scaled", shape=(1,)))
      elif column_name in features.INT_COLUMNS:
        # add a categorical_column for column_name + "_log_int"
        columns.append(tf.feature_column.numeric_column(
          column_name+"_log_01scaled", shape=(1,)))
      
      for column in columns:
        wide_columns.append(column)
        deep_columns.append(column)
      continue
    if column_name in features.EMBEDDING_DIMENSIONS:
      column = tf.feature_column.embedding_column(
        categorical_column,
        dimension=features.EMBEDDING_DIMENSIONS[column_name],
        combiner='mean')
    else:
      column = tf.feature_column.indicator_column(categorical_column)
    deep_columns.append(column)
  tf.compat.v1.logging.warn('deep columns: {}'.format(len(deep_columns)))
  tf.compat.v1.logging.warn('wide columns: {}'.format(len(wide_columns)))
  tf.compat.v1.logging.warn('wide&deep intersection: {}'.format(len(set(wide_columns).intersection(set(deep_columns)))))
  return wide_columns, deep_columns

def separate_input_fn(
    tf_transform_output,
    transformed_examples,
    create_batches,
    mode,
    reader_num_threads=1,
    parser_num_threads=2,
    shuffle_buffer_size=10,
    prefetch_buffer_size=1,
    print_display_ids=False):
  """
  A version of the training + eval input function that uses dataset operations.
  (For more straightforward tweaking.)
  """
  
  tf.compat.v1.logging.warn('Shuffle buffer size: {}'.format(shuffle_buffer_size))

  filenames_dataset = tf.data.Dataset.list_files(transformed_examples, shuffle=False)
  
  raw_dataset = tf.data.TFRecordDataset(filenames_dataset, 
            num_parallel_reads=reader_num_threads)
  
  raw_dataset = raw_dataset.shuffle(shuffle_buffer_size) \
                  if (mode==tf.estimator.ModeKeys.TRAIN and shuffle_buffer_size > 1) \
                  else raw_dataset
  raw_dataset = raw_dataset.repeat()
  raw_dataset = raw_dataset.batch(create_batches)
  
  # this function appears to require each element to be a vector
  # batching should mean that this is always true
  # one possible alternative for any problematic case is tf.io.parse_single_example
  parsed_dataset = raw_dataset.apply(tf.data.experimental.parse_example_dataset(
            tf_transform_output.transformed_feature_spec(), 
            num_parallel_calls=parser_num_threads))
  
  # a function mapped over each dataset element
  # will separate label, ensure that elements are two-dimensional (batch size, elements per record)
  # adds print_display_ids injection
  def consolidate_batch(elem):
    label = elem.pop('label')
    reshaped_label = tf.reshape(label, [-1, label.shape[-1]])
    reshaped_elem = {key: tf.reshape(elem[key], [-1, elem[key].shape[-1]]) for key in elem}
    if print_display_ids:
      elem['ad_id'] = tf.Print(input_=elem['ad_id'], 
        data=[tf.reshape(elem['display_id'], [-1])], 
        message='display_id', name='print_display_ids', summarize=FLAGS.eval_batch_size)
      elem['ad_id'] = tf.Print(input_=elem['ad_id'], 
        data=[tf.reshape(elem['ad_id'], [-1])],
        message='ad_id', name='print_ad_ids', summarize=FLAGS.eval_batch_size)
      elem['ad_id'] = tf.Print(input_=elem['ad_id'], 
        data=[tf.reshape(elem['is_leak'], [-1])],
        message='is_leak', name='print_is_leak', summarize=FLAGS.eval_batch_size)

    return reshaped_elem, reshaped_label
  
  if mode == tf.estimator.ModeKeys.EVAL:
    parsed_dataset = parsed_dataset.map(consolidate_batch, num_parallel_calls=None)
  else:
    parsed_dataset = parsed_dataset.map(consolidate_batch, 
      num_parallel_calls=parser_num_threads)
    parsed_dataset = parsed_dataset.prefetch(prefetch_buffer_size)

  return parsed_dataset

# rough approximation for MAP metric for measuring ad quality
# roughness comes from batch sizes falling between groups of
# display ids
# hack because of name clashes. Probably makes sense to rename features
DISPLAY_ID_COLUMN = features.DISPLAY_ID_COLUMN
def map_custom_metric(features, labels, predictions):
  display_ids = tf.reshape(features[DISPLAY_ID_COLUMN], [-1])
  predictions = predictions['probabilities'][:, 1]
  labels = labels[:, 0]

  # Processing unique display_ids, indexes and counts
  # Sorting needed in case the same display_id occurs in two different places
  sorted_ids = tf.argsort(display_ids)
  display_ids = tf.gather(display_ids, indices=sorted_ids)
  predictions = tf.gather(predictions, indices=sorted_ids)
  labels = tf.gather(labels, indices=sorted_ids)

  _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(
    display_ids, out_idx=tf.int64)
  pad_length = 30 - tf.reduce_max(display_ids_ads_count)
  pad_fn = lambda x: tf.pad(x, [(0, 0), (0, pad_length)])
 
  preds = tf.RaggedTensor.from_value_rowids(
    predictions, display_ids_idx).to_tensor()
  labels = tf.RaggedTensor.from_value_rowids(
    labels, display_ids_idx).to_tensor()

  labels = tf.argmax(labels, axis=1)

  return {
    'map': tf.compat.v1.metrics.average_precision_at_k(
      predictions=pad_fn(preds),
      labels=labels,
      k=12, 
      name="streaming_map")}


IS_LEAK_COLUMN = features.IS_LEAK_COLUMN
def map_custom_metric_with_leak(features, labels, predictions):
  display_ids = features[DISPLAY_ID_COLUMN]
  display_ids = tf.reshape(display_ids, [-1])
  is_leak_tf = features[IS_LEAK_COLUMN]
  is_leak_tf = tf.reshape(is_leak_tf, [-1])

  predictions = predictions['probabilities'][:, 1]
  predictions = predictions + tf.cast(is_leak_tf, tf.float32)
  labels = labels[:, 0]

  # Processing unique display_ids, indexes and counts
  # Sorting needed in case the same display_id occurs in two different places
  sorted_ids = tf.argsort(display_ids)
  display_ids = tf.gather(display_ids, indices=sorted_ids)
  predictions = tf.gather(predictions, indices=sorted_ids)
  labels = tf.gather(labels, indices=sorted_ids)

  _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(
    display_ids, out_idx=tf.int64)
  pad_length = 30 - tf.reduce_max(display_ids_ads_count)
  pad_fn = lambda x: tf.pad(x, [(0, 0), (0, pad_length)])

  preds = tf.RaggedTensor.from_value_rowids(predictions, display_ids_idx).to_tensor()
  labels = tf.RaggedTensor.from_value_rowids(labels, display_ids_idx).to_tensor()
  labels = tf.argmax(labels, axis=1)

  return {
    'map_with_leak': tf.compat.v1.metrics.average_precision_at_k(
      predictions=pad_fn(preds),
      labels=labels,
      k=12,
      name="streaming_map_with_leak")}

# A quick custom hook to access and log info about an Estimator graph
class InternalLoggerHook(tf.estimator.SessionRunHook):
  def __init__(self, logfile='internal_log.txt'):
    self.logfile = logfile
  
  def after_create_session(self, session, coord): # runs once, after graph is finalized
    log_writer = open(self.logfile, 'w')
    
    # one pass through to record a dictionary with {input: output} pairs of nodes
    # doesn't add much overhead that I can see, and makes it easier to trace dependencies
    hold_outputs_dict = {}
    for op in tf.get_default_graph().get_operations():
      for tensor in op.inputs:
        base_name = ''.join(tensor.name.split(':')[:-1]) if ':' in tensor.name else tensor.name
        if base_name not in hold_outputs_dict:
          hold_outputs_dict[base_name] = [op.name]
        else:
          hold_outputs_dict[base_name].append(op.name)
    
    # record information for each op to file, this time, drawing on the above dictionary for outputs
    for op in tf.get_default_graph().get_operations():
      op_repr = ''
      op_repr = op_repr + repr(op.node_def) # protobuf-style representation
      outputs = hold_outputs_dict.pop(op.name, [])
      op_repr = op_repr + '\n'.join(['output: ' + repr(o) for o in outputs] + \
                                    ['colocation_group: ' + repr(cg) for cg in op.colocation_groups()] + \
                                    ['control_input: ' + repr(ci) for ci in op.control_inputs])
      op_repr = '  ' + '\n  '.join(op_repr.split('\n')) # indented
      op_repr = op.name + ' {\n' + op_repr + '\n}\n\n'
      log_writer.write(op_repr)
    
    # leave a warning at the end of the file if any outputs are left over
    log_writer.write('Unclaimed outputs:\n' + '\n'.join([key + ': ' + repr(hold_outputs_dict[key]) \
                                                         for key in hold_outputs_dict]))
    
    log_writer.close()

# function to create a wide & deep Estimator, with options to knock out parts (model_type)
def custom_estimator_model_fn(features, labels, mode, params, config):
  
  with tf.compat.v1.variable_scope('deep', values=features) as scope:
    deep_absolute_scope = scope.name
    if params['model_type'] in [DEEP, WIDE_N_DEEP]:
      deep_current = tf.compat.v1.feature_column.input_layer(features, params['deep_columns'])

    if params['model_type'] in [DEEP, WIDE_N_DEEP]:
      for layer_ind in range(len(params['layers'])):
        num_neurons = params['layers'][layer_ind]
        deep_current = tf.keras.layers.Dense(num_neurons, activation=tf.nn.relu)(deep_current)
        deep_current = tf.keras.layers.Dropout(params['deep_dropout'])(deep_current)

      deep_logits = tf.keras.layers.Dense(1)(deep_current)
    else:
      deep_logits = None

  with tf.compat.v1.variable_scope('wide', values=features) as scope:
    wide_absolute_scope = scope.name
    wide_logits = tf.compat.v1.feature_column.linear_model(features, params['wide_columns'], units=1, sparse_combiner='sum') \
          if params['model_type'] in [WIDE, WIDE_N_DEEP] else None

  if deep_logits is None and wide_logits is None: # with only the input pipeline, just return input features
    assert mode == tf.estimator.ModeKeys.PREDICT, \
          'Only the input pipeline is used; eval and train don\'t have meaning'
    return tf.estimator.EstimatorSpec(mode, predictions=features)
  else:
    logits = deep_logits if wide_logits is None else wide_logits if deep_logits is None \
             else (wide_logits + deep_logits)

  head = tf.contrib.estimator.binary_classification_head(loss_reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE)

  def train_op_fn(loss):
    global_step = tf.compat.v1.train.get_global_step()
    deep_optimizer = params['deep_optimizer']
    wide_optimizer = params['wide_optimizer']

    # enable mixed precision if desired
    if params['amp']:
      deep_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(deep_optimizer)
      wide_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(wide_optimizer)
   
    deep_op = deep_optimizer.minimize(loss, var_list=tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=deep_absolute_scope)) if deep_logits is not None else None
    wide_op = wide_optimizer.minimize(loss, var_list=tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=wide_absolute_scope)) if wide_logits is not None else None
    train_op = tf.group(deep_op, wide_op) if deep_logits is not None and wide_logits is not None \
               else deep_op if deep_logits is not None else wide_op
    with tf.control_dependencies([train_op]): # increment global step once train op is done
      return tf.compat.v1.assign_add(global_step, 1).op # this is how the canned estimator appears to do it

  return head.create_estimator_spec(features, mode, logits, labels=labels, train_op_fn=train_op_fn)


# a helper function to create an estimator of the specified type, either custom or canned
# custom estimators are created with the custom_estimator_model_fn, and have some options working
# that the canned ones do not (AMP, knocking out parts of the model, NVTX)
def construct_estimator(model_type, custom_estimator, run_config, 
                        wide_columns, wide_optimizer, 
                        deep_columns, deep_hidden_units, deep_dropout, deep_optimizer, 
                        amp=False):
  if custom_estimator:
    estimator = tf.estimator.Estimator(
    model_fn=custom_estimator_model_fn,
    config=run_config,
    params={
        'wide_columns': wide_columns,
        'deep_columns': deep_columns,
        'deep_dropout': deep_dropout,
        'model_type': model_type,
        'layers': deep_hidden_units,
        'wide_optimizer': wide_optimizer,
        'deep_optimizer': deep_optimizer,
        'amp': amp,
    })
  else:
    assert model_type in [WIDE, DEEP, WIDE_N_DEEP], 'Canned estimator only supports basic wide, deep, wnd'
    assert not amp, 'AMP not functional for canned estimator' # AMP does not optimize the separate graph
    if model_type == WIDE:
      estimator = tf.estimator.LinearClassifier(
          feature_columns=wide_columns,
          config=run_config,
          optimizer=wide_optimizer)
   
    elif model_type == DEEP:
      estimator = tf.estimator.DNNClassifier(
          feature_columns=deep_columns,
          hidden_units=deep_hidden_units,
          dropout=deep_dropout,
          config=run_config,
          optimizer=deep_optimizer)
   
    elif model_type == WIDE_N_DEEP:
      estimator = tf.estimator.DNNLinearCombinedClassifier(
          config=run_config,
          linear_feature_columns=wide_columns,
          linear_optimizer=wide_optimizer,
          dnn_feature_columns=deep_columns,
          dnn_optimizer=deep_optimizer,
          dnn_hidden_units=deep_hidden_units,
          dnn_dropout=deep_dropout,
          linear_sparse_combiner='sum',
          loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    
  return estimator
   


def main(FLAGS):
  if FLAGS.hvd:
    hvd.init()
    if hvd.local_rank() == 0:
      tf.logging.set_verbosity(tf.logging.INFO)
      log_path = os.path.join(FLAGS.results_dir, FLAGS.log_filename)
      os.makedirs(FLAGS.results_dir, exist_ok=True)
      dllogger.init(backends=[
        dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
        	filename=log_path),
        dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
    else:
      tf.logging.set_verbosity(tf.logging.ERROR)
      dllogger.init(backends=[])
    num_gpus = hvd.size()
  else:
    tf.logging.set_verbosity(tf.logging.INFO)
    log_path = os.path.join(FLAGS.results_dir, FLAGS.log_filename)
    os.makedirs(FLAGS.results_dir, exist_ok=True)
    dllogger.init(backends=[
      dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
        filename=log_path),
      dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
    num_gpus = 1

  dllogger.log(data=vars(FLAGS), step='PARAMETER')
  
  local_batch_size = FLAGS.global_batch_size // num_gpus
  create_batches = local_batch_size // FLAGS.prebatch_size

  wide_columns, deep_columns = get_feature_columns(use_all_columns=FLAGS.use_all_columns)
  tf_transform_output = tft.TFTransformOutput(FLAGS.transformed_metadata_path)

  if not FLAGS.hvd or hvd.local_rank() == 0:
    tf.compat.v1.logging.warn('command line arguments: {}'.format(json.dumps(vars(FLAGS))))
    if not os.path.exists(FLAGS.results_dir):
      os.mkdir(FLAGS.results_dir)

    with open('{}/args.json'.format(FLAGS.results_dir), 'w') as f:
      json.dump(vars(FLAGS), f, indent=4)

  if FLAGS.gpu:
    session_config = tf.compat.v1.ConfigProto(log_device_placement=FLAGS.log_device_placement)
  else:
    session_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}, log_device_placement=FLAGS.log_device_placement)
  
  if FLAGS.hvd:
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())

  if FLAGS.xla:
    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  
  if FLAGS.benchmark:
    model_dir = None
  else:
    model_dir = FLAGS.model_dir

  if FLAGS.save_checkpoints_steps != 0:
    run_config = tf.estimator.RunConfig(model_dir=model_dir).replace(session_config=session_config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=1)
  else:
    run_config = tf.estimator.RunConfig(model_dir=model_dir).replace(session_config=session_config,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      keep_checkpoint_max=1)

  wide_optimizer = tf.compat.v1.train.FtrlOptimizer(
    learning_rate=FLAGS.linear_learning_rate,
    l1_regularization_strength=FLAGS.linear_l1_regularization,
    l2_regularization_strength=FLAGS.linear_l2_regularization)

  deep_optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(
    learning_rate=FLAGS.deep_learning_rate,
    initial_accumulator_value=0.1,
    l1_regularization_strength=FLAGS.deep_l1_regularization,
    l2_regularization_strength=FLAGS.deep_l2_regularization,
    use_locking=False)
  
  if FLAGS.hvd:
    wide_optimizer = hvd.DistributedOptimizer(wide_optimizer)
    deep_optimizer = hvd.DistributedOptimizer(deep_optimizer)

  stats_filename = os.path.join(FLAGS.transformed_metadata_path, 'stats.json')
  embed_columns = None
  
  # input functions to read data from disk
  train_input_fn = lambda : separate_input_fn(
    tf_transform_output,
    FLAGS.train_data_pattern,
    create_batches,
    tf.estimator.ModeKeys.TRAIN,
    reader_num_threads=FLAGS.reader_num_threads,
    parser_num_threads=FLAGS.parser_num_threads,
    shuffle_buffer_size=int(FLAGS.shuffle_percentage*create_batches),
    prefetch_buffer_size=FLAGS.prefetch_buffer_size,
    print_display_ids=FLAGS.print_display_ids)
  eval_input_fn = lambda : separate_input_fn(
    tf_transform_output,
    FLAGS.eval_data_pattern,
    (FLAGS.eval_batch_size // FLAGS.prebatch_size),
    tf.estimator.ModeKeys.EVAL,
    reader_num_threads=1,
    parser_num_threads=1,
    shuffle_buffer_size=int(FLAGS.shuffle_percentage*create_batches),
    prefetch_buffer_size=FLAGS.prefetch_buffer_size,
    print_display_ids=FLAGS.print_display_ids)
  
  estimator = construct_estimator(FLAGS.model_type, not FLAGS.canned_estimator, run_config, 
                                  wide_columns, wide_optimizer,
                                  deep_columns, FLAGS.deep_hidden_units, FLAGS.deep_dropout, deep_optimizer, 
                                  amp=FLAGS.amp)
  
  estimator = tf.estimator.add_metrics(estimator, map_custom_metric)
  estimator = tf.estimator.add_metrics(estimator, map_custom_metric_with_leak)

  steps_per_epoch = FLAGS.training_set_size / FLAGS.global_batch_size

  print('Steps per epoch: {}'.format(steps_per_epoch))
  max_steps = int(FLAGS.num_epochs * steps_per_epoch)

  hooks = []
  if FLAGS.hvd:
    hooks.append(hvd.BroadcastGlobalVariablesHook(0))

  if FLAGS.predict or FLAGS.evaluate: # inference
    if FLAGS.benchmark:
      benchmark_hook = BenchmarkLoggingHook(global_batch_size=FLAGS.eval_batch_size, warmup_steps=FLAGS.benchmark_warmup_steps)
      hooks.append(benchmark_hook)
      eval_steps = FLAGS.benchmark_steps
    else:
      eval_steps = FLAGS.eval_steps

    predict_result_iter = estimator.predict(input_fn=eval_input_fn, hooks=hooks, yield_single_examples=False)
    
    results = []
    for i, r in enumerate(predict_result_iter):
      print('predicting batch: ', i)
      results.append(r)
      # TODO: use eval_steps
      if i >= eval_steps - 1:
        break

    if FLAGS.benchmark:
      infer_throughput = benchmark_hook.mean_throughput.value()
      
    if FLAGS.benchmark:
      dllogger.log(data={'infer_throughput': infer_throughput}, step=tuple())
    elif FLAGS.evaluate:
      print('evaluating using estimator.evaluate with eval_batch_size = ', 
        FLAGS.eval_batch_size, ' and eval_steps = ', FLAGS.eval_steps)

      result = estimator.evaluate(eval_input_fn, hooks=hooks, steps=FLAGS.eval_steps)
      dllogger.log(step=(), data={'map_infer': float(result['map']), 'map_with_leak_infer': float(result['map_with_leak'])})
    elif FLAGS.predict:
      scores = [r['probabilities'][:, 1] for r in results]
      scores = np.hstack(scores)
      scores_path = os.path.join(FLAGS.model_dir, 'scores.txt')
      print('saving the numpy scores array to: ', scores_path)
      np.savetxt(scores_path, scores, fmt="%f", delimiter='\n')

  else: # training

    if FLAGS.benchmark:
      benchmark_hook = BenchmarkLoggingHook(global_batch_size=FLAGS.global_batch_size, 
        warmup_steps=FLAGS.benchmark_warmup_steps)
      hooks.append(benchmark_hook)
      estimator.train(train_input_fn, hooks=hooks, steps=FLAGS.benchmark_steps)
      train_throughput = benchmark_hook.mean_throughput.value()
      dllogger.log(data={'train_throughput': train_throughput}, step=tuple())
    else:
      train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps, hooks=hooks)
      eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      throttle_secs=FLAGS.eval_throttle_secs, steps=FLAGS.eval_steps)  
      result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

      if result != (None, None):
        dllogger.log(step=(), data={'map': float(result[0]['map']), 
        'map_with_leak': float(result[0]['map_with_leak'])})
    
    
if __name__ == '__main__':
  FLAGS = create_parser().parse_args()
  main(FLAGS)

