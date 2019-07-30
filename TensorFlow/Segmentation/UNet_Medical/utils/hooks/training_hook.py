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

from dllogger import LOGGER, tags


class TrainingHook(tf.train.SessionRunHook):

    def __init__(self, log_every=1):
        self._log_every = log_every
        self._iter_idx = 0

    def before_run(self, run_context):
        run_args = tf.train.SessionRunArgs(
            fetches=[
                'cross_loss_ref:0',
                'dice_loss_ref:0',
                'total_loss_ref:0',
            ]
        )

        return run_args

    def after_run(self,
                  run_context,
                  run_values):
        cross_loss, dice_loss, total_loss = run_values.results

        if self._iter_idx % self._log_every == 0:
            LOGGER.log('cross_loss', cross_loss)
            LOGGER.log('dice_loss', dice_loss)
            LOGGER.log('total_loss', total_loss)
        self._iter_idx += 1
