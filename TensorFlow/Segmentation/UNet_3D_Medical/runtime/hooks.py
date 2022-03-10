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

""" Hooks for metric collection and benchmarking """
import time

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd


def get_hooks(params, logger):
    """ Get the appropriate set of hooks given the configuration

    :param params: Dict with additional parameters
    :param logger: Logger object
    :return: Set of hooks
    """

    hooks = []

    if params.exec_mode == 'debug_train':
        return get_debug_training_hooks(logger, params)

    if params.exec_mode == 'debug_predict':
        return get_debug_predict_hooks(logger, params)

    if 'train' in params.exec_mode:
        return get_training_hooks(logger, params)

    if params.exec_mode == 'predict':
        return get_predict_hooks(logger, params)

    return hooks


def get_debug_predict_hooks(logger, params):
    """ Return hooks for debugging prediction

    :param logger: Logger object
    :param params: Dict with additional parameters
    :return: Estimator hooks
    """
    hooks = []
    if hvd.rank() == 0:
        hooks += [ProfilingHook(warmup_steps=params.warmup_steps,
                                global_batch_size=params.batch_size,
                                logger=logger,
                                mode='inference')]
    return hooks


def get_debug_training_hooks(logger, params):
    """ Return hooks for debugging training

    :param logger: Logger object
    :param params: Dict with additional parameters
    :return: Estimator hooks
    """
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    if hvd.rank() == 0:
        hooks += [TrainingHook(log_every=params.log_every,
                               logger=logger,
                               tensor_names=['total_loss_ref:0']),
                  ProfilingHook(warmup_steps=params.warmup_steps,
                                global_batch_size=hvd.size() * params.batch_size,
                                logger=logger,
                                mode='train')]
    return hooks


def get_predict_hooks(logger, params):
    """ Return hooks for prediction

    :param logger: Logger object
    :param params: Dict with additional parameters
    :return: Estimator hooks
    """
    hooks = []

    if hvd.rank() == 0:
        if params.benchmark:
            hooks = [ProfilingHook(warmup_steps=params.warmup_steps,
                                   global_batch_size=params.batch_size,
                                   logger=logger,
                                   mode='test')]
    return hooks


def get_training_hooks(logger, params):
    """ Return hooks for training

    :param logger: Logger object
    :param params: Dict with additional parameters
    :return: Estimator hooks
    """
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    if hvd.rank() == 0:
        hooks += [OomReportingHook()]

        if params.benchmark:
            hooks += [ProfilingHook(warmup_steps=params.warmup_steps,
                                    global_batch_size=hvd.size() * params.batch_size,
                                    logger=logger,
                                    mode='train')]
        else:
            hooks += [TrainingHook(log_every=params.log_every,
                                   logger=logger,
                                   tensor_names=['total_loss_ref:0'])]

    return hooks


class ProfilingHook(tf.estimator.SessionRunHook):
    """ Hook for profiling metrics """

    def __init__(self, warmup_steps, global_batch_size, logger, mode):
        """ Build hook

        :param warmup_steps: Number of steps to skip initially
        :param global_batch_size: Number of samples per bach in all gpus
        :param logger: Logger object
        :param mode: Estimator's execution mode
        """
        self._warmup_steps = warmup_steps
        self._global_batch_size = global_batch_size
        self._step = 0
        self._timestamps = []
        self._logger = logger
        self._mode = mode

    def before_run(self, _):
        """ Execute before run """
        self._step += 1
        if self._step >= self._warmup_steps:
            self._timestamps.append(time.time())

    def end(self, _):
        """ Execute on completion """
        deltas = np.array([self._timestamps[i + 1] - self._timestamps[i] for i in range(len(self._timestamps) - 1)])
        stats = process_performance_stats(np.array(deltas),
                                          self._global_batch_size,
                                          self._mode)

        self._logger.log(step=(), data=stats)
        self._logger.flush()


class TrainingHook(tf.estimator.SessionRunHook):
    """ Hook for training metrics """

    def __init__(self, log_every, logger, tensor_names):
        """ Build hook for training

        :param log_every: Logging frequency
        :param logger: Logger object
        :param tensor_names: Names of the tensors to log
        """
        self._log_every = log_every
        self._step = 0
        self._logger = logger
        self._tensor_names = tensor_names

    def before_run(self, _):
        """ Execute before run """
        run_args = tf.compat.v1.train.SessionRunArgs(
            fetches=self._tensor_names
        )

        return run_args

    def after_run(self,
                  _,
                  run_values):
        """ Execute after run

        :param run_values: Values to capture
        :return:
        """
        if self._step % self._log_every == 0:
            for i in range(len(self._tensor_names)):
                self._logger.log(step=(self._step,), data={self._tensor_names[i]: str(run_values.results[i])})
        self._step += 1

    def end(self, _):
        """ Execute on completion """
        self._logger.flush()


class OomReportingHook(tf.estimator.SessionRunHook):  # pylint: disable=R0903
    """ Report for out of memory errors"""

    def before_run(self, _):  # pylint: disable=R0201
        """ Execute before run """
        return tf.estimator.SessionRunArgs(fetches=[],  # no extra fetches
                                           options=tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True))


def process_performance_stats(timestamps, batch_size, mode):
    """ Get confidence intervals

    :param timestamps: Collection of timestamps
    :param batch_size: Number of samples per batch
    :param mode: Estimator's execution mode
    :return: Stats
    """
    timestamps_ms = 1000 * timestamps
    throughput_imgps = (1000.0 * batch_size / timestamps_ms).mean()
    stats = {f"throughput_{mode}": throughput_imgps,
             f"latency_{mode}_mean": timestamps_ms.mean()}
    for level in [90, 95, 99]:
        stats.update({f"latency_{mode}_{level}": np.percentile(timestamps_ms, level)})

    return stats
