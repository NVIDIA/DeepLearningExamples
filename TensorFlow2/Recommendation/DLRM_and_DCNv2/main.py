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
#
# author: Tomasz Grel (tgrel@nvidia.com)


from absl import app, flags
import os
import sys
import json
from distributed_embeddings.python.layers import dist_model_parallel as dmp


# Define the flags first before importing TensorFlow.
# Otherwise, enabling XLA-Lite would be impossible with a command-line flag
def define_common_flags():
    flags.DEFINE_enum("mode", default="train", enum_values=['inference', 'eval', 'train'],
                      help='Choose "train" to train the model, "inference" to benchmark inference'
                      ' and "eval" to run validation')
    # Debug parameters
    flags.DEFINE_bool("run_eagerly", default=False, help="Disable all tf.function decorators for debugging")
    
    flags.DEFINE_bool("tfdata_debug", default=False, help="Run tf.data operations eagerly (experimental)")

    flags.DEFINE_integer("seed", default=None, help="Random seed")

    flags.DEFINE_bool("embedding_zeros_initializer", default=False,
                      help="Initialize the embeddings to zeros. This takes much less time so it's useful"
                      " for benchmarking and debugging.")

    flags.DEFINE_bool("embedding_trainable", default=True, help="If True the embeddings will be trainable, otherwise frozen")

    # Hardware and performance features
    flags.DEFINE_bool("amp", default=False, help="Enable automatic mixed precision")
    flags.DEFINE_bool("use_mde_embeddings", default=True,
                      help="Use the embedding implementation from the TensorFlow Distributed Embeddings package")
    flags.DEFINE_bool("concat_embedding", default=False,
                      help="Concatenate embeddings with the same dimension. Only supported for singleGPU.")
    flags.DEFINE_string("dist_strategy", default='memory_balanced',
                        help="Strategy for the Distributed Embeddings to use. Supported options are"
                        "'memory_balanced', 'basic' and 'memory_optimized'")
    flags.DEFINE_integer("column_slice_threshold", default=5*1000*1000*1000,
                         help='Number of elements above which a distributed embedding will be sliced across'
                         'multiple devices')
    flags.DEFINE_integer("row_slice_threshold", default=10*1000*1000*1000,
                         help='Number of elements above which a distributed embedding will be sliced across'
                         'multiple devices')
    flags.DEFINE_integer("data_parallel_threshold", default=None,
                         help='Number of elements above which a distributed embedding will be sliced across'
                         'multiple devices')

    flags.DEFINE_integer("cpu_offloading_threshold_gb", default=75,
                         help='Size of the embedding tables in GB above which '
                              'offloading to CPU memory should be employed.'
                              'Applies only to singleGPU at the moment.')

    flags.DEFINE_bool('cpu', default=False, help='Place the entire model on CPU')

    flags.DEFINE_bool("xla", default=False, help="Enable XLA")

    flags.DEFINE_integer("loss_scale", default=65536, help="Static loss scale to use with mixed precision training")

    flags.DEFINE_integer("inter_op_parallelism", default=None, help='Number of inter op threads')
    flags.DEFINE_integer("intra_op_parallelism", default=None, help='Number of intra op threads')

    # Checkpointing
    flags.DEFINE_string("save_checkpoint_path", default=None,
                        help="Path to which to save a checkpoint file at the end of the training")
    flags.DEFINE_string("restore_checkpoint_path", default=None,
                        help="Path from which to restore a checkpoint before training")

    # Evaluation, logging, profiling
    flags.DEFINE_integer("auc_thresholds", default=8000,
                         help="Number of thresholds for the AUC computation")

    flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train for")
    flags.DEFINE_integer("max_steps", default=-1, help="Stop the training/inference after this many optimiation steps")

    flags.DEFINE_integer("evals_per_epoch", default=1, help='Number of evaluations per epoch')
    flags.DEFINE_float("print_freq", default=100, help='Number of steps between debug prints')

    flags.DEFINE_integer("profiler_start_step", default=None, help='Step at which to start profiling')
    flags.DEFINE_integer("profiled_rank", default=1, help='Rank to profile')

    flags.DEFINE_string("log_path", default='dlrm_tf_log.json', help="Path to JSON file for storing benchmark results")

    # dataset and dataloading settings
    flags.DEFINE_string("dataset_path", default=None,
                        help="Path to dataset directory")
    flags.DEFINE_string("feature_spec", default="feature_spec.yaml",
                        help="Name of the feature spec file in the dataset directory")
    flags.DEFINE_enum("dataset_type", default="tf_raw",
                      enum_values=['tf_raw', 'synthetic', 'split_tfrecords'],
                      help='The type of the dataset to use')
    flags.DEFINE_boolean("data_parallel_input", default=False, help="Use a data-parallel dataloader,"
                         " i.e., load a local batch of of data for all input features")

    # Synthetic dataset settings
    flags.DEFINE_boolean("synthetic_dataset_use_feature_spec", default=False,
                         help="Create a temporary synthetic dataset based on a real one. "
                              "Uses --dataset_path and --feature_spec"
                              "Overrides synthetic dataset dimension flags, except the number of batches")
    flags.DEFINE_integer('synthetic_dataset_train_batches', default=64008,
                         help='Number of training batches in the synthetic dataset')
    flags.DEFINE_integer('synthetic_dataset_valid_batches', default=1350,
                         help='Number of validation batches in the synthetic dataset')
    flags.DEFINE_list('synthetic_dataset_cardinalities', default=26*[1000],
                         help='Number of categories for each embedding table of the synthetic dataset')
    flags.DEFINE_list('synthetic_dataset_hotness', default=26*[20],
                         help='Number of categories for each embedding table of the synthetic dataset')
    flags.DEFINE_integer('synthetic_dataset_num_numerical_features', default=13,
                         help='Number of numerical features of the synthetic dataset')

define_common_flags()

FLAGS = flags.FLAGS
app.define_help_flags()
app.parse_flags_with_usage(sys.argv)

if FLAGS.xla:
    if FLAGS.cpu:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible --tf_xla_cpu_global_jit'
    else:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'


import time
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import horovod.tensorflow as hvd
from tensorflow.keras.mixed_precision import LossScaleOptimizer

import dllogger

from utils.logging import IterTimer, init_logging
from utils.distributed import dist_print
from dataloading.dataloader import create_input_pipelines, get_dataset_metadata
from nn.lr_scheduler import LearningRateScheduler
from nn.model import Model
from nn.evaluator import Evaluator
from nn.trainer import Trainer


def init_tf(FLAGS):
    """
    Set global options for TensorFlow
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    visible_gpus = []
    if gpus and not FLAGS.cpu:
        visible_gpus = gpus[hvd.local_rank()]
    tf.config.experimental.set_visible_devices(visible_gpus, 'GPU')

    if FLAGS.amp:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    tf.config.run_functions_eagerly(FLAGS.run_eagerly)

    if FLAGS.tfdata_debug:
        tf.data.experimental.enable_debug_mode()

    if FLAGS.inter_op_parallelism:
        tf.config.threading.set_inter_op_parallelism_threads(FLAGS.inter_op_parallelism)

    if FLAGS.intra_op_parallelism:
        tf.config.threading.set_intra_op_parallelism_threads(FLAGS.intra_op_parallelism)

    tf.random.set_seed(hash((FLAGS.seed, hvd.rank())))


def parse_embedding_dimension(embedding_dim, num_embeddings):
    try:
        embedding_dim = int(embedding_dim)
        embedding_dim = [embedding_dim] * num_embeddings
        return embedding_dim
    except:
        pass

    if not isinstance(embedding_dim, str):
        return ValueError(f'Unsupported embedding_dimension type: f{type(embedding_dim)}')

    if os.path.exists(embedding_dim):
        # json file with a list of dimensions for each feature
        with open(embedding_dim) as f:
            edim = json.load(f)
    else:
        edim = embedding_dim.split(',')

    edim = [int(d) for d in edim]

    if len(edim) != num_embeddings:
        raise ValueError(f'Length of specified embedding dimensions ({len(edim)}) does not match'
                         f' the number of embedding layers in the neural network ({num_embeddings})')

    return edim


def compute_eval_points(train_batches, evals_per_epoch):
    eval_points = np.linspace(0, train_batches - 1, evals_per_epoch + 1)[1:]
    eval_points = np.round(eval_points).tolist()
    return eval_points


def inference_benchmark(validation_pipeline, dlrm, timer, FLAGS):
    if FLAGS.max_steps == -1:
        FLAGS.max_steps = 1000

    evaluator = Evaluator(model=dlrm, timer=timer, auc_thresholds=FLAGS.auc_thresholds,
                          max_steps=FLAGS.max_steps, cast_dtype=None)

    auc, test_loss, latencies = evaluator(validation_pipeline)

    # don't benchmark the first few warmup steps
    latencies = latencies[10:]
    result_data = {
        'mean_inference_throughput': FLAGS.valid_batch_size / np.mean(latencies),
        'mean_inference_latency': np.mean(latencies)
    }

    for percentile in [90, 95, 99]:
        result_data[f'p{percentile}_inference_latency'] = np.percentile(latencies, percentile)
    result_data['auc'] = auc

    if hvd.rank() == 0:
        dllogger.log(data=result_data, step=tuple())


def validate_cmd_line_flags():
    if FLAGS.cpu and hvd.size() > 1:
        raise ValueError('MultiGPU mode is not supported when training on CPU')

    if FLAGS.cpu and FLAGS.interaction == 'custom_cuda':
        raise ValueError('"custom_cuda" dot interaction not supported for CPU. '
        'Please specify "--dot_interaction tensorflow" if you want to run on CPU')

    if FLAGS.concat_embedding and hvd.size() != 1:
        raise ValueError('Concat embedding is currently unsupported in multiGPU mode.')

    if FLAGS.concat_embedding and FLAGS.dataset_type != 'tf_raw':
        raise ValueError('Concat embedding is only supported for dataset_type="tf_raw",'
                         f'got dataset_type={FLAGS.dataset_type}')

    all_embedding_dims_equal = all(dim == FLAGS.embedding_dim[0] for dim in FLAGS.embedding_dim)
    if FLAGS.concat_embedding and not all_embedding_dims_equal:
        raise ValueError('Concat embedding is only supported when all embeddings have the same output dimension,'
                         f'got embedding_dim={FLAGS.embedding_dim}')


def create_optimizers(flags):
    if flags.optimizer == 'sgd':
        embedding_optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=flags.learning_rate, momentum=0)
        if flags.amp:
            embedding_optimizer = LossScaleOptimizer(embedding_optimizer,
                                                     initial_scale=flags.loss_scale,
                                                     dynamic=False)
        mlp_optimizer = embedding_optimizer

    elif flags.optimizer == 'adam':
        embedding_optimizer = tfa.optimizers.LazyAdam(learning_rate=flags.learning_rate,
                                                      beta_1=flags.beta1, beta_2=flags.beta2)

        mlp_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=flags.learning_rate,
                                                 beta_1=flags.beta1, beta_2=flags.beta2)
        if flags.amp:
            # only wrap the mlp optimizer and not the embedding optimizer because the embeddings are not run in FP16
            mlp_optimizer = LossScaleOptimizer(mlp_optimizer, initial_scale=flags.loss_scale, dynamic=False)

    return mlp_optimizer, embedding_optimizer


def main():
    hvd.init()
    init_logging(log_path=FLAGS.log_path, params_dict=FLAGS.flag_values_dict(), enabled=hvd.rank()==0)
    init_tf(FLAGS)

    dataset_metadata = get_dataset_metadata(FLAGS.dataset_path, FLAGS.feature_spec)

    FLAGS.embedding_dim = parse_embedding_dimension(FLAGS.embedding_dim,
                                                    num_embeddings=len(dataset_metadata.categorical_cardinalities))

    validate_cmd_line_flags()

    if FLAGS.restore_checkpoint_path is not None:
        model = Model.create_from_checkpoint(FLAGS.restore_checkpoint_path)
    else:
        model = Model(**FLAGS.flag_values_dict(), num_numerical_features=dataset_metadata.num_numerical_features,
                      categorical_cardinalities=dataset_metadata.categorical_cardinalities,
                      transpose=False)

    table_ids = model.sparse_model.get_local_table_ids(hvd.rank())
    print(f'local feature ids={table_ids}')

    train_pipeline, validation_pipeline = create_input_pipelines(dataset_type=FLAGS.dataset_type,
                                                                 dataset_path=FLAGS.dataset_path,
                                                                 train_batch_size=FLAGS.batch_size,
                                                                 test_batch_size=FLAGS.valid_batch_size,
                                                                 table_ids=table_ids,
                                                                 feature_spec=FLAGS.feature_spec,
                                                                 rank=hvd.rank(), world_size=hvd.size(),
                                                                 concat_features=FLAGS.concat_embedding,
                                                                 data_parallel_input=FLAGS.data_parallel_input)

    mlp_optimizer, embedding_optimizer = create_optimizers(FLAGS)

    scheduler = LearningRateScheduler([mlp_optimizer, embedding_optimizer],
                                      warmup_steps=FLAGS.warmup_steps,
                                      base_lr=FLAGS.learning_rate,
                                      decay_start_step=FLAGS.decay_start_step,
                                      decay_steps=FLAGS.decay_steps)

    timer = IterTimer(train_batch_size=FLAGS.batch_size, test_batch_size=FLAGS.batch_size,
                      optimizer=embedding_optimizer, print_freq=FLAGS.print_freq, enabled=hvd.rank() == 0)

    if FLAGS.mode == 'inference':
        inference_benchmark(validation_pipeline, model, timer, FLAGS)
        return
    elif FLAGS.mode == 'eval':
        evaluator = Evaluator(model=model, timer=timer, auc_thresholds=FLAGS.auc_thresholds, max_steps=FLAGS.max_steps)
        test_auc, test_loss, _ = evaluator(validation_pipeline)

        if hvd.rank() == 0:
            dllogger.log(data=dict(auc=test_auc, test_loss=test_loss), step=tuple())
        return

    eval_points = compute_eval_points(train_batches=len(train_pipeline),
                                      evals_per_epoch=FLAGS.evals_per_epoch)

    trainer = Trainer(model, embedding_optimizer=embedding_optimizer, mlp_optimizer=mlp_optimizer, amp=FLAGS.amp,
                      lr_scheduler=scheduler, tf_dataset_op=train_pipeline.op, cpu=FLAGS.cpu)

    evaluator = Evaluator(model=model, timer=timer, auc_thresholds=FLAGS.auc_thresholds, distributed=hvd.size() > 1)

    best_auc = 0
    best_loss = 1e6
    train_begin = time.time()
    for epoch in range(FLAGS.epochs):
        print('Starting epoch: ', epoch)
        for step in range(len(train_pipeline)):
            if step == FLAGS.profiler_start_step and hvd.rank() == FLAGS.profiled_rank:
                tf.profiler.experimental.start('logdir')

            if FLAGS.profiler_start_step and step == FLAGS.profiler_start_step + 100 and hvd.rank() == FLAGS.profiled_rank:
                tf.profiler.experimental.stop()

            loss = trainer.train_step()

            if step == 0 and hvd.size() > 1:
                dmp.broadcast_variables(model.variables, root_rank=0)

            if step % FLAGS.print_freq == 0:
                if tf.math.is_nan(loss):
                    print('NaN loss encountered in training. Aborting.')
                    break

            timer.step_train(loss=loss)

            if FLAGS.max_steps != -1 and step > FLAGS.max_steps:
                dist_print(f'Max steps of {FLAGS.max_steps} reached, exiting')
                break

            if step in eval_points:
                test_auc, test_loss, _ = evaluator(validation_pipeline)
                dist_print(f'Evaluation completed, AUC: {test_auc:.6f}, test_loss: {test_loss:.6f}')
                timer.test_idx = 0
                best_auc = max(best_auc, test_auc)
                best_loss = min(best_loss, test_loss)

    elapsed = time.time() - train_begin

    if FLAGS.save_checkpoint_path is not None:
        model.save_checkpoint(FLAGS.save_checkpoint_path)

    if hvd.rank() == 0:
        dist_print(f'Training run completed, elapsed: {elapsed:.0f} [s]')
        results = {
            'throughput': FLAGS.batch_size / timer.mean_train_time(),
            'mean_step_time_ms': timer.mean_train_time() * 1000,
            'auc': best_auc,
            'validation_loss': best_loss
        }
        dllogger.log(data=results, step=tuple())
