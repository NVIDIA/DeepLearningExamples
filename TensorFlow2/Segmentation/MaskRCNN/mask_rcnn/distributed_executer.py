#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""Interface to run mask rcnn model in different distributed strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import six

import math

import multiprocessing

import tensorflow as tf

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils.distributed_utils import MPI_is_distributed
from mask_rcnn.utils.distributed_utils import MPI_local_rank
from mask_rcnn.utils.distributed_utils import MPI_rank

from mask_rcnn.hooks.logging_hook import AutoLoggingHook

from mask_rcnn.utils.lazy_imports import LazyImport
hvd = LazyImport("horovod.tensorflow")

from tensorflow.core.protobuf import rewriter_config_pb2

from mask_rcnn import evaluation
from mask_rcnn.hyperparameters import params_io
from mask_rcnn.hooks import CheckpointSaverHook
from mask_rcnn.hooks import PretrainedWeightsLoadingHook


def get_training_hooks(mode, model_dir, checkpoint_path=None, skip_checkpoint_variables=None):

    assert mode in ('train', 'eval')

    training_hooks = [
        AutoLoggingHook(
            # log_every_n_steps=RUNNING_CONFIG.display_step,
            log_every_n_steps=5 if "NGC_JOB_ID" not in os.environ else 100,
            # warmup_steps=RUNNING_CONFIG.warmup_steps,
            warmup_steps=100,
            is_training=True
        )
    ]

    if not MPI_is_distributed() or MPI_rank() == 0:
        training_hooks.append(PretrainedWeightsLoadingHook(
            prefix="resnet50/",
            checkpoint_path=checkpoint_path,
            skip_variables_regex=skip_checkpoint_variables
        ))

    if MPI_is_distributed() and mode == "train":
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(root_rank=0))

    if not MPI_is_distributed() or MPI_rank() == 0:
        training_hooks.append(CheckpointSaverHook(
            checkpoint_dir=model_dir,
            checkpoint_basename="model.ckpt"
        ))

    return training_hooks


@six.add_metaclass(abc.ABCMeta)
class BaseExecuter(object):
  """Interface to run Mask RCNN model in TPUs/GPUs.

  Arguments:
    flags: FLAGS object passed from the user.
    model_config: Model configuration needed to run distribution strategy.
    model_fn: Model function to be passed to Estimator.
  """

  def __init__(self, runtime_config, model_fn):

    self._runtime_config = runtime_config
    self._model_fn = model_fn

    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

  @staticmethod
  def _get_session_config(mode, use_xla, use_amp, use_tf_distributed=False, allow_xla_at_inference=False):

      assert mode in ('train', 'eval')

      rewrite_options = rewriter_config_pb2.RewriterConfig(
          # arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          # arithmetic_optimization=rewriter_config_pb2.RewriterConfig.ON,

          # constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
          # constant_folding=rewriter_config_pb2.RewriterConfig.ON,           # TO TEST

          # debug_stripper=rewriter_config_pb2.RewriterConfig.OFF,
          # debug_stripper=rewriter_config_pb2.RewriterConfig.ON,           # TO TEST

          # dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          # dependency_optimization=rewriter_config_pb2.RewriterConfig.ON,           # TO TEST

          # disable_model_pruning=False,           # INCOMPATIBLE with AMP
          # function_optimization=True,
          # implementation_selector=True,

          # loop_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          # loop_optimization=rewriter_config_pb2.RewriterConfig.ON,           # TO TEST

          # The default setting (SCHEDULING and SWAPPING HEURISTICS only)
          # memory_optimization=rewriter_config_pb2.RewriterConfig.DEFAULT_MEM_OPT,

          # Disabled in the meta-optimizer.
          # memory_optimization=rewriter_config_pb2.RewriterConfig.NO_MEM_OPT,

          # Driven by manual op-level annotations.
          # memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL,

          # Swapping heuristic will move a tensor from the GPU to the CPU and move it
          # back when needed to reduce peak memory usage..
          # memory_optimization=rewriter_config_pb2.RewriterConfig.SWAPPING_HEURISTICS,

          # Recomputation heuristics will recompute ops (such as Relu activation)
          # during backprop instead of storing them, reducing peak memory usage.
          # memory_optimization=rewriter_config_pb2.RewriterConfig.RECOMPUTATION_HEURISTICS,

          # Scheduling will split big ops such as AddN and try to enforce a schedule of
          # the new computations that decreases peak memory usage.
          # memory_optimization=rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS,

          # Use any combination of swapping and recomputation heuristics.
          # memory_optimization=rewriter_config_pb2.RewriterConfig.HEURISTICS,

          meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.TWO,
          # meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE,
          # meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.DEFAULT_NUM_ITERS,

          # pin_to_host_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          # pin_to_host_optimization=rewriter_config_pb2.RewriterConfig.ON,         # TO TEST
          #
          # remapping=rewriter_config_pb2.RewriterConfig.OFF,
          # remapping=rewriter_config_pb2.RewriterConfig.ON,                   # TO TEST

          # scoped_allocator_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          # scoped_allocator_optimization=rewriter_config_pb2.RewriterConfig.ON,  # TO TEST

          # shape_optimization=rewriter_config_pb2.RewriterConfig.OFF,
          # shape_optimization=rewriter_config_pb2.RewriterConfig.ON,           # TO TEST
      )

      if use_amp:
          logging.info("[%s] AMP is activated - Experiment Feature" % mode)
          rewrite_options.auto_mixed_precision = True

      config = tf.compat.v1.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False,
          graph_options=tf.compat.v1.GraphOptions(
              rewrite_options=rewrite_options,
              # infer_shapes=True  # Heavily drops throughput by 30%
          )
      )

      if use_tf_distributed:
        config.gpu_options.force_gpu_compatible = False

      else:
        config.gpu_options.force_gpu_compatible = True  # Force pinned memory

        if MPI_is_distributed():
            config.gpu_options.visible_device_list = str(MPI_local_rank())

      if use_xla and (mode == "train" or allow_xla_at_inference):
          logging.info("[%s] XLA is activated - Experiment Feature" % mode)
          config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
          # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2

      if mode == 'train':
          config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads

          if MPI_is_distributed():
              config.inter_op_parallelism_threads = max(2, multiprocessing.cpu_count() // hvd.local_size())

          elif not use_tf_distributed:
              config.inter_op_parallelism_threads = 4

      return config

  @abc.abstractmethod
  def build_strategy_configuration(self, mode):
    """Builds run configuration for distributed train/eval.

    Returns:
      RunConfig with distribution strategy configurations
      to pass to the constructor of TPUEstimator/Estimator.
    """

    NotImplementedError('Must be implemented in subclass')

  def build_model_parameters(self, mode):
    """Builds model parameter."""

    assert mode in ('train', 'eval')

    batch_size = self._runtime_config.train_batch_size if mode == 'train' else self._runtime_config.eval_batch_size

    params = dict(
        self._runtime_config.values(),
        mode=mode,
        batch_size=batch_size,
        model_dir=self._runtime_config.model_dir,
    )

    if mode == 'eval':
      params = dict(
        params,
        augment_input_data=False,
      )

    return params

  def build_mask_rcnn_estimator(self, params, run_config, mode):
    """Creates TPUEstimator/Estimator instance.

    Arguments:
      params: A dictionary to pass to Estimator `model_fn`.
      run_config: RunConfig instance specifying distribution strategy
        configurations.
      mode: Mode -- one of 'train` or `eval`.

    Returns:
      TFEstimator or TPUEstimator instance.
    """
    assert mode in ('train', 'eval')

    return tf.estimator.Estimator(
        model_fn=self._model_fn,
        model_dir=self._runtime_config.model_dir,
        config=run_config,
        params=params
    )

  def _save_config(self):
    """Save parameters to config files if model_dir is defined."""

    model_dir = self._runtime_config.model_dir

    if model_dir is not None:
      if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)

      params_io.save_hparams_to_yaml(self._runtime_config, model_dir + '/params.yaml')

  def _write_summary(self, summary_dir, eval_results, predictions, current_step):

    if not self._runtime_config.visualize_images_summary:
      predictions = None

    evaluation.write_summary(eval_results, summary_dir, current_step, predictions=predictions)

  def train(self, train_input_fn, run_eval_after_train=False, eval_input_fn=None):
    """Run distributed training on Mask RCNN model."""

    self._save_config()
    train_run_config = self.build_strategy_configuration('train')
    train_params = self.build_model_parameters('train')
    train_estimator = self.build_mask_rcnn_estimator(train_params, train_run_config, 'train')

    train_estimator.train(
        input_fn=train_input_fn,
        max_steps=self._runtime_config.total_steps,
        hooks=get_training_hooks(
            mode="train",
            model_dir=self._runtime_config.model_dir,
            checkpoint_path=self._runtime_config.checkpoint,
            skip_checkpoint_variables=self._runtime_config.skip_checkpoint_variables
        )
    )

    if not run_eval_after_train:
      return None

    if eval_input_fn is None:
      raise ValueError('Eval input_fn must be passed to conduct evaluation after training.')

    eval_run_config = self.build_strategy_configuration('eval')
    eval_params = self.build_model_parameters('eval')
    eval_estimator = self.build_mask_rcnn_estimator(eval_params, eval_run_config, 'eval')

    last_ckpt = tf.train.latest_checkpoint(self._runtime_config.model_dir, latest_filename=None)
    logging.info("Restoring parameters from %s\n" % last_ckpt)

    eval_results, predictions = evaluation.evaluate(
        eval_estimator,
        eval_input_fn,
        self._runtime_config.eval_samples,
        self._runtime_config.eval_batch_size,
        self._runtime_config.include_mask,
        self._runtime_config.val_json_file,
        report_frequency=self._runtime_config.report_frequency
    )

    output_dir = os.path.join(self._runtime_config.model_dir, 'eval')
    tf.io.gfile.makedirs(output_dir)

    # Summary writer writes out eval metrics.
    self._write_summary(output_dir, eval_results, predictions, self._runtime_config.total_steps)

    return eval_results

  def train_and_eval(self, train_input_fn, eval_input_fn):
    """Run distributed train and eval on Mask RCNN model."""

    self._save_config()
    output_dir = os.path.join(self._runtime_config.model_dir, 'eval')
    tf.io.gfile.makedirs(output_dir)

    train_run_config = self.build_strategy_configuration('train')
    train_params = self.build_model_parameters('train')
    train_estimator = self.build_mask_rcnn_estimator(train_params, train_run_config, 'train')

    eval_estimator = None
    eval_results = None

    num_cycles = math.ceil(self._runtime_config.total_steps / self._runtime_config.num_steps_per_eval)

    training_hooks = get_training_hooks(
        mode="train",
        model_dir=self._runtime_config.model_dir,
        checkpoint_path=self._runtime_config.checkpoint,
        skip_checkpoint_variables=self._runtime_config.skip_checkpoint_variables
    )

    for cycle in range(1, num_cycles + 1):

      if not MPI_is_distributed() or MPI_rank() == 0:

        print()  # Visual Spacing
        logging.info("=================================")
        logging.info('     Start training cycle %02d' % cycle)
        logging.info("=================================\n")

      max_cycle_step = min(int(cycle * self._runtime_config.num_steps_per_eval), self._runtime_config.total_steps)

      PROFILER_ENABLED = False

      if (not MPI_is_distributed() or MPI_rank() == 0) and PROFILER_ENABLED:
          profiler_context_manager = tf.contrib.tfprof.ProfileContext

      else:
          from contextlib import suppress
          profiler_context_manager = lambda *args, **kwargs: suppress()  # No-Op context manager

      with profiler_context_manager(
              '/workspace/profiling/',
              trace_steps=range(100, 200, 3),
              dump_steps=[200]
      ) as pctx:

          if (not MPI_is_distributed() or MPI_rank() == 0) and PROFILER_ENABLED:
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory()
            pctx.add_auto_profiling('op', opts, [150, 200])

          train_estimator.train(
              input_fn=train_input_fn,
              max_steps=max_cycle_step,
              hooks=training_hooks,
          )

      if not MPI_is_distributed() or MPI_rank() == 0:

          print()  # Visual Spacing
          logging.info("=================================")
          logging.info('    Start evaluation cycle %02d' % cycle)
          logging.info("=================================\n")

          if eval_estimator is None:
              eval_run_config = self.build_strategy_configuration('eval')
              eval_params = self.build_model_parameters('eval')
              eval_estimator = self.build_mask_rcnn_estimator(eval_params, eval_run_config, 'eval')

          last_ckpt = tf.train.latest_checkpoint(self._runtime_config.model_dir, latest_filename=None)
          logging.info("Restoring parameters from %s\n" % last_ckpt)

          eval_results, predictions = evaluation.evaluate(
              eval_estimator,
              eval_input_fn,
              self._runtime_config.eval_samples,
              self._runtime_config.eval_batch_size,
              self._runtime_config.include_mask,
              self._runtime_config.val_json_file,
              report_frequency=self._runtime_config.report_frequency
          )

          self._write_summary(output_dir, eval_results, predictions, max_cycle_step)

      if MPI_is_distributed():
          from mpi4py import MPI
          MPI.COMM_WORLD.Barrier()  # Waiting for all MPI processes to sync

    return eval_results

  def eval(self, eval_input_fn):
    """Run distributed eval on Mask RCNN model."""

    output_dir = os.path.join(self._runtime_config.model_dir, 'eval')
    tf.io.gfile.makedirs(output_dir)

    # Summary writer writes out eval metrics.
    run_config = self.build_strategy_configuration('eval')
    eval_params = self.build_model_parameters('eval')
    eval_estimator = self.build_mask_rcnn_estimator(eval_params, run_config, 'eval')

    logging.info('Starting to evaluate.')

    last_ckpt = tf.train.latest_checkpoint(self._runtime_config.model_dir, latest_filename=None)

    if last_ckpt is not None:
      logging.info("Restoring parameters from %s\n" % last_ckpt)
      current_step = int(os.path.basename(last_ckpt).split('-')[1])

    else:
      logging.warning(
          "Could not find trained model in model_dir: `%s`, running initialization to predict\n" %
          self._runtime_config.model_dir
      )
      current_step = 0

    eval_results, predictions = evaluation.evaluate(
        eval_estimator,
        eval_input_fn,
        self._runtime_config.eval_samples,
        self._runtime_config.eval_batch_size,
        self._runtime_config.include_mask,
        self._runtime_config.val_json_file
    )

    self._write_summary(output_dir, eval_results, predictions, current_step)

    if current_step >= self._runtime_config.total_steps:
        logging.info('Evaluation finished after training step %d' % current_step)

    return eval_results


class EstimatorExecuter(BaseExecuter):
  """Interface that runs Mask RCNN model using TPUEstimator."""

  def __init__(self, runtime_config, model_fn):
    super(EstimatorExecuter, self).__init__(runtime_config, model_fn)

    if MPI_is_distributed():
      os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
      os.environ['HOROVOD_NUM_NCCL_STREAMS'] = '1'
      # os.environ['HOROVOD_AUTOTUNE'] = '2'

      hvd.init()

      logging.info("Horovod successfully initialized ...")

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1' if not MPI_is_distributed() else str(hvd.size())

    os.environ['TF_SYNC_ON_FINISH'] = '0'

  def build_strategy_configuration(self, mode):
    """Retrieves model configuration for running TF Estimator."""

    run_config = tf.estimator.RunConfig(
        tf_random_seed=(
            self._runtime_config.seed
            if not MPI_is_distributed() or self._runtime_config.seed is None else
            self._runtime_config.seed + MPI_rank()
        ),
        model_dir=self._runtime_config.model_dir,
        save_summary_steps=None,  # disabled
        save_checkpoints_steps=None,  # disabled
        save_checkpoints_secs=None,  # disabled
        keep_checkpoint_max=20,  # disabled
        keep_checkpoint_every_n_hours=None,  # disabled
        log_step_count_steps=None,  # disabled
        session_config=self._get_session_config(
            mode=mode,
            use_xla=self._runtime_config.use_xla,
            use_amp=self._runtime_config.use_amp,
            use_tf_distributed=False,
            allow_xla_at_inference=self._runtime_config.allow_xla_at_inference  # TODO: Remove when XLA at inference fixed
        ),
        protocol=None,
        device_fn=None,
        train_distribute=None,
        eval_distribute=None,
        experimental_distribute=None
    )

    return run_config


class TFDistributedExecuter(BaseExecuter):
  """Interface that runs Mask RCNN model using MultiWorkerMirroredStrategy."""

  @staticmethod
  def is_eval_task():
    return tf.distribute.cluster_resolver.TFConfigClusterResolver().task_type == 'evaluator'

  def build_strategy_configuration(self, mode):
    """Retrieves model configuration for MultiWorkerMirroredStrategy."""

    distributed_strategy = tf.distribute.MirroredStrategy()
    # distributed_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    #     tf.distribute.experimental.CollectiveCommunication.NCCL
    # )

    run_config = tf.estimator.RunConfig(
        tf_random_seed=self._runtime_config.seed,
        model_dir=self._runtime_config.model_dir,
        save_summary_steps=None,  # disabled
        save_checkpoints_steps=None,  # disabled
        save_checkpoints_secs=None,  # disabled
        keep_checkpoint_max=20,  # disabled
        keep_checkpoint_every_n_hours=None,  # disabled
        log_step_count_steps=None,  # disabled
        session_config=self._get_session_config(
            mode=mode,
            use_xla=self._runtime_config.use_xla,
            use_amp=self._runtime_config.use_amp,
            use_tf_distributed=True,
            # TODO: Remove when XLA at inference fixed
            allow_xla_at_inference=self._runtime_config.allow_xla_at_inference
        ),
        protocol=None,
        device_fn=None,
        train_distribute=distributed_strategy if mode == "train" else None,
        eval_distribute=None,
        experimental_distribute=None
    )

    return run_config
