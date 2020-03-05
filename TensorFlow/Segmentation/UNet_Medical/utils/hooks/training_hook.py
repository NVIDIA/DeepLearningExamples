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
import horovod.tensorflow as hvd


class TrainingHook(tf.estimator.SessionRunHook):

    def __init__(self, logger, max_steps, log_every=1):
        self._log_every = log_every
        self._iter_idx = 0
        self.logger = logger
        self.max_steps = max_steps

    def before_run(self, run_context):
        run_args = tf.estimator.SessionRunArgs(
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

        if (self._iter_idx % self._log_every == 0) and (hvd.rank() == 0):
            self.logger.log(step=(self._iter_idx, self.max_steps),
                            data={'train_ce_loss': float(cross_loss),
                                  'train_dice_loss': float(dice_loss),
                                  'train_total_loss': float(total_loss)})
        self._iter_idx += 1
