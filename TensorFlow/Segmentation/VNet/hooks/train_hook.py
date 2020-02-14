# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf

import dllogger as DLLogger


class TrainHook(tf.estimator.SessionRunHook):
    def __init__(self, log_every, logger):
        self._log_every = log_every
        self._step = 0
        self._logger = logger

    def before_run(self, run_context):
        run_args = tf.train.SessionRunArgs(
            fetches=[
                'vnet/loss/total_loss_ref:0',
            ]
        )

        return run_args

    def after_run(self,
                  run_context,
                  run_values):
        if self._step % self._log_every == 0:
            self._logger.log(step=(self._step,), data={'total_loss': str(run_values.results[0])})
        self._step += 1

    def end(self, session):
        self._logger.flush()
