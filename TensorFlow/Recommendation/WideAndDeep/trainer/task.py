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

import argparse
import dllogger
import horovod.tensorflow as hvd
import json
import numpy as np
import os
import tensorflow as tf
import tensorflow_transform as tft
from trainer import features
from utils.dataloader import separate_input_fn
from utils.hooks.benchmark_hooks import BenchmarkLoggingHook
from utils.metrics import map_custom_metric, map_custom_metric_with_leak
from utils.schedulers import learning_rate_scheduler

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
        '--train_data_pattern',
        help='Pattern of training file names. For example if training files are train_000.tfrecord, \
    train_001.tfrecord then --train_data_pattern is train_*',
        type=str,
        default='/outbrain/tfrecords/train/part*',
        nargs='+')
    parser.add_argument(
        '--eval_data_pattern',
        help='Pattern of eval file names. For example if eval files are eval_000.tfrecord, \
    eval_001.tfrecord then --eval_data_pattern is eval_*',
        type=str,
        default='/outbrain/tfrecords/eval/part*',
        nargs='+')
    parser.add_argument(
        '--model_dir',
        help='Model Checkpoint will be saved here',
        type=str,
        default='/outbrain/checkpoints')
    parser.add_argument(
        '--transformed_metadata_path',
        help='Path to transformed_metadata.',
        type=str,
        default='/outbrain/tfrecords')
    parser.add_argument(
        '--deep_hidden_units',
        help='Hidden units per layer, separated by spaces',
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
        '--eval_epoch_interval',
        help='Perform evaluation during training after this many epochs',
        default=2,
        type=float)
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
        '--deep_warmup_epochs',
        help='Number of epochs for deep LR warmup',
        type=float,
        default=0)
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
        '--shuffle_percentage',
        type=float,
        default=0.0,
        help='Size of the shuffle buffer from 0 to 1. \
        1 means that the shuffle buffer size will be equal to the size of the entire batch.')
    parser.add_argument(
        '--print_display_ids',
        help='Print the display ids processed by the input pipeline',
        default=False,
        action='store_true')
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


def construct_estimator(model_type, run_config,
                        wide_columns, wide_optimizer,
                        deep_columns, deep_hidden_units, deep_dropout, deep_optimizer):
    assert model_type in [WIDE, DEEP, WIDE_N_DEEP], 'Canned estimator only supports basic wide, deep, wnd'
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

    wide_columns, deep_columns = features.get_feature_columns()
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
        session_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0},
                                                  log_device_placement=FLAGS.log_device_placement)

    if FLAGS.hvd:
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    if FLAGS.xla:
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    if FLAGS.benchmark:
        model_dir = None
    else:
        model_dir = FLAGS.model_dir

    steps_per_epoch = FLAGS.training_set_size / FLAGS.global_batch_size

    print('Steps per epoch: {}'.format(steps_per_epoch))
    max_steps = int(FLAGS.num_epochs * steps_per_epoch)

    run_config = tf.estimator.RunConfig(model_dir=model_dir) \
        .replace(session_config=session_config,
                 save_checkpoints_steps=int(FLAGS.eval_epoch_interval * steps_per_epoch),
                 keep_checkpoint_max=1)

    def wide_optimizer():
        opt = tf.compat.v1.train.FtrlOptimizer(
            learning_rate=FLAGS.linear_learning_rate,
            l1_regularization_strength=FLAGS.linear_l1_regularization,
            l2_regularization_strength=FLAGS.linear_l2_regularization)
        if FLAGS.hvd:
            opt = hvd.DistributedOptimizer(opt)
        if FLAGS.amp:
            opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        return opt

    def deep_optimizer():
        with tf.device("/cpu:0"):
            learning_rate_fn = learning_rate_scheduler(
                lr_init=FLAGS.deep_learning_rate,
                warmup_steps=int(steps_per_epoch * FLAGS.deep_warmup_epochs),
                global_step=tf.compat.v1.train.get_global_step()
            )
        opt = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=learning_rate_fn,
            initial_accumulator_value=0.1,
            use_locking=False)
        if FLAGS.hvd:
            opt = hvd.DistributedOptimizer(opt)
        if FLAGS.amp:
            opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        return opt

    # input functions to read data from disk
    train_input_fn = lambda: separate_input_fn(
        tf_transform_output,
        FLAGS.train_data_pattern,
        create_batches,
        tf.estimator.ModeKeys.TRAIN,
        reader_num_threads=FLAGS.reader_num_threads,
        parser_num_threads=FLAGS.parser_num_threads,
        shuffle_buffer_size=int(FLAGS.shuffle_percentage * create_batches),
        prefetch_buffer_size=FLAGS.prefetch_buffer_size,
        print_display_ids=FLAGS.print_display_ids)
    eval_input_fn = lambda: separate_input_fn(
        tf_transform_output,
        FLAGS.eval_data_pattern,
        (FLAGS.eval_batch_size // FLAGS.prebatch_size),
        tf.estimator.ModeKeys.EVAL,
        reader_num_threads=1,
        parser_num_threads=1,
        shuffle_buffer_size=int(FLAGS.shuffle_percentage * create_batches),
        prefetch_buffer_size=FLAGS.prefetch_buffer_size,
        print_display_ids=FLAGS.print_display_ids)

    estimator = construct_estimator(FLAGS.model_type, run_config,
                                    wide_columns, wide_optimizer,
                                    deep_columns, FLAGS.deep_hidden_units, FLAGS.deep_dropout, deep_optimizer)

    estimator = tf.estimator.add_metrics(estimator, map_custom_metric)
    estimator = tf.estimator.add_metrics(estimator, map_custom_metric_with_leak)

    hooks = []
    if FLAGS.hvd:
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    if FLAGS.predict or FLAGS.evaluate:  # inference
        if FLAGS.benchmark:
            benchmark_hook = BenchmarkLoggingHook(global_batch_size=FLAGS.eval_batch_size,
                                                  warmup_steps=FLAGS.benchmark_warmup_steps)
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
            dllogger.log(step=(), data={'map_infer': float(result['map']),
                                        'map_with_leak_infer': float(result['map_with_leak'])})
        elif FLAGS.predict:
            scores = [r['probabilities'][:, 1] for r in results]
            scores = np.hstack(scores)
            scores_path = os.path.join(FLAGS.model_dir, 'scores.txt')
            print('saving the numpy scores array to: ', scores_path)
            np.savetxt(scores_path, scores, fmt="%f", delimiter='\n')

    else:  # training

        if FLAGS.benchmark:
            benchmark_hook = BenchmarkLoggingHook(global_batch_size=FLAGS.global_batch_size,
                                                  warmup_steps=FLAGS.benchmark_warmup_steps)
            hooks.append(benchmark_hook)
            estimator.train(train_input_fn, hooks=hooks, steps=FLAGS.benchmark_steps)
            train_throughput = benchmark_hook.mean_throughput.value()
            dllogger.log(data={'train_throughput': train_throughput}, step=tuple())
        else:
            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                                max_steps=max_steps,
                                                hooks=hooks)
            eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                              throttle_secs=0,
                                              steps=FLAGS.eval_steps)
            result = tf.estimator.train_and_evaluate(estimator=estimator,
                                                     train_spec=train_spec,
                                                     eval_spec=eval_spec)

            if result != (None, None):
                dllogger.log(step=(), data={'map': float(result[0]['map']),
                                            'map_with_leak': float(result[0]['map_with_leak'])})


if __name__ == '__main__':
    FLAGS = create_parser().parse_args()
    main(FLAGS)
