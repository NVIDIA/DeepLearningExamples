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
from distributed_embeddings.python.layers import dist_model_parallel as dmp

# Define the flags first before importing TensorFlow.
# Otherwise, enabling XLA-Lite would be impossible with a command-line flag
def define_command_line_flags():
    flags.DEFINE_enum("mode", default="train", enum_values=['inference', 'eval', 'train', 'deploy'],
                      help='Choose "train" to train the model, "inference" to benchmark inference'
                      ' and "eval" to run validation')
    flags.DEFINE_float("learning_rate", default=24, help="Learning rate")
    flags.DEFINE_integer("batch_size", default=64 * 1024, help="Batch size used for training")
    flags.DEFINE_bool("run_eagerly", default=False, help="Disable all tf.function decorators for debugging")

    flags.DEFINE_bool("dummy_model", default=False, help="Use a dummy model for benchmarking and debugging")

    flags.DEFINE_list("top_mlp_dims", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
    flags.DEFINE_list("bottom_mlp_dims", [512, 256, 128], "Linear layer sizes for the bottom MLP")

    flags.DEFINE_enum("optimizer", default="sgd", enum_values=['sgd', 'adam'],
                      help='The optimization algorithm to be used.')

    flags.DEFINE_string("save_checkpoint_path", default=None,
                        help="Path to which to save a checkpoint file at the end of the training")
    flags.DEFINE_string("restore_checkpoint_path", default=None,
                        help="Path from which to restore a checkpoint before training")

    flags.DEFINE_string("saved_model_output_path", default=None,
                        help='Path for storing the model in TensorFlow SavedModel format')
    flags.DEFINE_bool("save_input_signature", default=False,
                      help="Save input signature in the SavedModel")
    flags.DEFINE_string("saved_model_input_path", default=None,
                        help='Path for loading the model in TensorFlow SavedModel format')

    flags.DEFINE_bool('cpu', default=False, help='Place the entire model on CPU')

    flags.DEFINE_bool("amp", default=False, help="Enable automatic mixed precision")
    flags.DEFINE_bool("fp16", default=False,
                      help="Create the model in pure FP16 precision, suitable only for inference and deployment")
    flags.DEFINE_bool("xla", default=False, help="Enable XLA")

    flags.DEFINE_integer("loss_scale", default=1024, help="Static loss scale to use with mixed precision training")

    flags.DEFINE_integer("auc_thresholds", default=8000,
                         help="Number of thresholds for the AUC computation")

    flags.DEFINE_integer("epochs", default=1, help="Number of epochs to train for")
    flags.DEFINE_integer("max_steps", default=-1, help="Stop the training/inference after this many optimiation steps")

    flags.DEFINE_bool("embedding_trainable", default=True, help="If True the embeddings will be trainable, otherwise frozen")

    flags.DEFINE_enum("dot_interaction", default="custom_cuda", enum_values=["custom_cuda", "tensorflow", "dummy"],
                      help="Dot interaction implementation to use")

    flags.DEFINE_integer("embedding_dim", default=128, help='Number of columns in the embedding tables')

    flags.DEFINE_integer("evals_per_epoch", default=1, help='Number of evaluations per epoch')
    flags.DEFINE_float("print_freq", default=100, help='Number of steps between debug prints')

    flags.DEFINE_integer("warmup_steps", default=8000,
                        help='Number of steps over which to linearly increase the LR at the beginning')
    flags.DEFINE_integer("decay_start_step", default=48000, help='Optimization step at which to start the poly LR decay')
    flags.DEFINE_integer("decay_steps", default=24000, help='Number of steps over which to decay from base LR to 0')

    flags.DEFINE_integer("profiler_start_step", default=None, help='Step at which to start profiling')
    flags.DEFINE_integer("profiled_rank", default=1, help='Rank to profile')

    flags.DEFINE_integer("inter_op_parallelism", default=None, help='Number of inter op threads')
    flags.DEFINE_integer("intra_op_parallelism", default=None, help='Number of intra op threads')

    flags.DEFINE_string("dist_strategy", default='memory_balanced',
                        help="Strategy for the Distributed Embeddings to use. Supported options are"
                        "'memory_balanced', 'basic' and 'memory_optimized'")

    flags.DEFINE_bool("use_merlin_de_embeddings", default=False,
                      help="Use the embedding implementation from the TensorFlow Distributed Embeddings package")


    flags.DEFINE_integer("column_slice_threshold", default=10*1000*1000*1000,
                         help='Number of elements above which a distributed embedding will be sliced across'
                         'multiple devices')

    flags.DEFINE_string("log_path", default='dlrm_tf_log.json', help="Path to JSON file for storing benchmark results")

    #dataset and dataloading settings
    flags.DEFINE_string("dataset_path", default=None,
                        help="Path to dataset directory")
    flags.DEFINE_string("feature_spec", default="feature_spec.yaml",
                        help="Name of the feature spec file in the dataset directory")
    flags.DEFINE_enum("dataset_type", default="tf_raw", enum_values=['tf_raw', 'synthetic'],
                      help='The type of the dataset to use')

    # Synthetic dataset settings
    flags.DEFINE_boolean("synthetic_dataset_use_feature_spec", default=False,
                         help="Create a temporary synthetic dataset based on a real one. "
                              "Uses --dataset_path and --feature_spec"
                              "Overrides synthetic dataset dimension flags, other than the number of batches")
    flags.DEFINE_integer('synthetic_dataset_train_batches', default=64008,
                         help='Number of training batches in the synthetic dataset')
    flags.DEFINE_integer('synthetic_dataset_valid_batches', default=1350,
                         help='Number of validation batches in the synthetic dataset')
    flags.DEFINE_list('synthetic_dataset_cardinalities', default=26*[1000],
                         help='Number of categories for each embedding table of the synthetic dataset')
    flags.DEFINE_integer('synthetic_dataset_num_numerical_features', default=13,
                         help='Number of numerical features of the synthetic dataset')

define_command_line_flags()

FLAGS = flags.FLAGS
app.define_help_flags()
app.parse_flags_with_usage(sys.argv)

if FLAGS.xla:
    if FLAGS.cpu:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible --tf_xla_cpu_global_jit'
    else:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'


import time
from lr_scheduler import LearningRateScheduler
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from utils import IterTimer, init_logging, dist_print
from dataloader import create_input_pipelines, get_dataset_metadata
from model import Dlrm, DummyDlrm, DlrmTrainer, evaluate
import horovod.tensorflow as hvd
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import dllogger


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

    if FLAGS.fp16:
        policy = tf.keras.mixed_precision.Policy("float16")
        tf.keras.mixed_precision.experimental.set_global_policy(policy)

    tf.config.run_functions_eagerly(FLAGS.run_eagerly)

    if FLAGS.inter_op_parallelism:
        tf.config.threading.set_inter_op_parallelism_threads(FLAGS.inter_op_parallelism)

    if FLAGS.intra_op_parallelism:
        tf.config.threading.set_intra_op_parallelism_threads(FLAGS.intra_op_parallelism)


def compute_eval_points(train_batches, evals_per_epoch):
    eval_points = np.linspace(0, train_batches - 1, evals_per_epoch + 1)[1:]
    eval_points = np.round(eval_points).tolist()
    return eval_points


def inference_benchmark(validation_pipeline, dlrm, timer, FLAGS):
    if FLAGS.max_steps == -1:
        FLAGS.max_steps = 1000

    if FLAGS.saved_model_input_path:
        cast_dtype = tf.float16 if FLAGS.amp else tf.float32
    else:
        cast_dtype = None

    auc, test_loss, latencies = evaluate(validation_pipeline, dlrm,
                               timer, auc_thresholds=FLAGS.auc_thresholds,
                               max_steps=FLAGS.max_steps, cast_dtype=cast_dtype)

    # don't benchmark the first few warmup steps
    latencies = latencies[10:]
    result_data = {
        'mean_inference_throughput': FLAGS.batch_size / np.mean(latencies),
        'mean_inference_latency': np.mean(latencies)
    }

    for percentile in [90, 95, 99]:
        result_data[f'p{percentile}_inference_latency'] = np.percentile(latencies, percentile)
    result_data['auc'] = auc

    if hvd.rank() == 0:
        dllogger.log(data=result_data, step=tuple())


def validate_cmd_line_flags():

    if FLAGS.restore_checkpoint_path is not None and FLAGS.saved_model_input_path is not None:
        raise ValueError('Incompatible cmd-line flags.'
                         'You can only specify one of --restore_checkpoint_path'
                         'and --saved_model_input_path at a time.')

    if FLAGS.saved_model_input_path is not None and FLAGS.mode == 'train':
        raise ValueError('Training from a SavedModel is not supported.'
                         'To train from a checkpoint please specify the '
                         '--restore_checkpoint_path cmd-line flag.')

    if FLAGS.cpu and hvd.size() > 1:
        raise ValueError('MultiGPU mode is not supported when training on CPU')

    if FLAGS.cpu and FLAGS.dot_interaction == 'custom_cuda':
        raise ValueError('"custom_cuda" dot interaction not supported for CPU. '
        'Please specify "--dot_interaction tensorflow" if you want to run on CPU')

    if FLAGS.fp16 and FLAGS.amp:
        raise ValueError('Only one of --amp and --fp16 can be specified at a time.')


def main(argv):
    hvd.init()
    validate_cmd_line_flags()
    init_logging(log_path=FLAGS.log_path, FLAGS=FLAGS)
    init_tf(FLAGS)

    dataset_metadata = get_dataset_metadata(FLAGS)
    dlrm = Dlrm.load_model_if_path_exists(FLAGS.saved_model_input_path)
    if dlrm is None:
        if FLAGS.dummy_model:
            dlrm = DummyDlrm(FLAGS=FLAGS, dataset_metadata=dataset_metadata)
        else:
            dlrm = Dlrm(FLAGS=FLAGS, dataset_metadata=dataset_metadata)
            dlrm = dlrm.restore_checkpoint_if_path_exists(FLAGS.restore_checkpoint_path)

    train_pipeline, validation_pipeline = create_input_pipelines(FLAGS, dlrm.local_table_ids)

    if FLAGS.optimizer == 'sgd':
        embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate, momentum=0)
        if FLAGS.amp:
            embedding_optimizer = LossScaleOptimizer(embedding_optimizer,
                                                     initial_scale=FLAGS.loss_scale,
                                                     dynamic=False)
        mlp_optimizer = embedding_optimizer
        optimizers = [mlp_optimizer]

    elif FLAGS.optimizer == 'adam':
        embedding_optimizer = tfa.optimizers.LazyAdam(learning_rate=FLAGS.learning_rate)
        mlp_optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
        if FLAGS.amp:
            embedding_optimizer = LossScaleOptimizer(embedding_optimizer,
                                                     initial_scale=FLAGS.loss_scale,
                                                     dynamic=False)
            mlp_optimizer = LossScaleOptimizer(mlp_optimizer,
                                               initial_scale=FLAGS.loss_scale,
                                               dynamic=False)
        optimizers = [mlp_optimizer, embedding_optimizer]

    scheduler = LearningRateScheduler(optimizers,
                                      warmup_steps=FLAGS.warmup_steps,
                                      base_lr=FLAGS.learning_rate,
                                      decay_start_step=FLAGS.decay_start_step,
                                      decay_steps=FLAGS.decay_steps)

    timer = IterTimer(train_batch_size=FLAGS.batch_size, test_batch_size=FLAGS.batch_size,
                      optimizer=embedding_optimizer, print_freq=FLAGS.print_freq, enabled=hvd.rank() == 0)


    if FLAGS.mode == 'inference':
        inference_benchmark(validation_pipeline, dlrm, timer, FLAGS)
        return
    elif FLAGS.mode == 'deploy':
        dlrm.save_model_if_path_exists(FLAGS.saved_model_output_path,
                                       save_input_signature=FLAGS.save_input_signature)
        print('deployed to: ', FLAGS.saved_model_output_path)
        return

    elif FLAGS.mode == 'eval':
        test_auc, test_loss, _ = evaluate(validation_pipeline, dlrm,
                                          timer, auc_thresholds=FLAGS.auc_thresholds)
        if hvd.rank() == 0:
            dllogger.log(data=dict(auc=test_auc, test_loss=test_loss), step=tuple())
        return

    eval_points = compute_eval_points(train_batches=len(train_pipeline),
                                      evals_per_epoch=FLAGS.evals_per_epoch)

    trainer = DlrmTrainer(dlrm, embedding_optimizer=embedding_optimizer,
                          mlp_optimizer=mlp_optimizer, amp=FLAGS.amp,
                          lr_scheduler=scheduler,
                          pipe=train_pipeline, cpu=FLAGS.cpu)

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
                dmp.broadcast_variables(trainer.dlrm.variables, root_rank=0)

            if step % 100 == 0:
                if tf.math.is_nan(loss):
                    print('NaN loss encountered in training. Aborting.')
                    break

            timer.step_train(loss=loss)

            if FLAGS.max_steps != -1 and step > FLAGS.max_steps:
                dist_print(f'Max steps of {FLAGS.max_steps} reached, exiting')
                break

            if step in eval_points:
                test_auc, test_loss, _ = evaluate(validation_pipeline, dlrm, timer, FLAGS.auc_thresholds)
                dist_print(f'Evaluation completed, AUC: {test_auc:.6f}, test_loss: {test_loss:.6f}')
                timer.test_idx = 0
                best_auc = max(best_auc, test_auc)
                best_loss = min(best_loss, test_loss)

    elapsed = time.time() - train_begin
    dlrm.save_checkpoint_if_path_exists(FLAGS.save_checkpoint_path)
    dlrm.save_model_if_path_exists(FLAGS.saved_model_output_path,
                                   save_input_signature=FLAGS.save_input_signature)

    if hvd.rank() == 0:
        dist_print(f'Training run completed, elapsed: {elapsed:.0f} [s]')
        results = {
            'throughput': FLAGS.batch_size / timer.mean_train_time(),
            'mean_step_time_ms': timer.mean_train_time() * 1000,
            'auc': best_auc,
            'validation_loss': best_loss,
            'train_loss': loss.numpy().item()
        }
        dllogger.log(data=results, step=tuple())


if __name__ == '__main__':
    app.run(main)
