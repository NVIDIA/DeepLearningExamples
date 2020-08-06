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

import time

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd


def get_hooks(params, logger):
    if 'train' in params.exec_mode:
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        if hvd.rank() == 0:
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

    elif 'predict' == params.exec_mode:
        hooks = []
        if hvd.rank() == 0:
            if params.benchmark:
                hooks += [ProfilingHook(warmup_steps=params.warmup_steps,
                                        global_batch_size=params.batch_size,
                                        logger=logger,
                                        mode='test')]
            return hooks


class ProfilingHook(tf.estimator.SessionRunHook):
    def __init__(self, warmup_steps, global_batch_size, logger, mode):
        self._warmup_steps = warmup_steps
        self._global_batch_size = global_batch_size
        self._step = 0
        self._timestamps = []
        self._logger = logger
        self._mode = mode

    def before_run(self, run_context):
        self._step += 1
        if self._step >= self._warmup_steps:
            self._timestamps.append(time.time())

    def end(self, session):
        deltas = np.array([self._timestamps[i + 1] - self._timestamps[i] for i in range(len(self._timestamps) - 1)])
        stats = process_performance_stats(np.array(deltas),
                                          self._global_batch_size,
                                          self._mode)

        self._logger.log(step=(), data={metric: float(value) for (metric, value) in stats})
        self._logger.flush()


class TrainingHook(tf.estimator.SessionRunHook):
    def __init__(self, log_every, logger, tensor_names):
        self._log_every = log_every
        self._step = 0
        self._logger = logger
        self._tensor_names = tensor_names

    def before_run(self, run_context):
        run_args = tf.train.SessionRunArgs(
            fetches=self._tensor_names
        )

        return run_args

    def after_run(self,
                  run_context,
                  run_values):
        if self._step % self._log_every == 0:
            for i in range(len(self._tensor_names)):
                self._logger.log(step=(self._step,), data={self._tensor_names[i]: str(run_values.results[i])})
        self._step += 1

    def end(self, session):
        self._logger.flush()


def process_performance_stats(timestamps, batch_size, mode):
    timestamps_ms = 1000 * timestamps
    latency_ms = timestamps_ms.mean()
    std = timestamps_ms.std()
    n = np.sqrt(len(timestamps_ms))
    throughput_imgps = (1000.0 * batch_size / timestamps_ms).mean()

    stats = [("throughput_{}".format(mode), str(throughput_imgps)),
             ('latency_{}:'.format(mode), str(latency_ms))]
    for ci, lvl in zip(["90%:", "95%:", "99%:"],
                       [1.645, 1.960, 2.576]):
        stats.append(("Latency_{} ".format(mode) + ci, str(latency_ms + lvl * std / n)))
    return stats
