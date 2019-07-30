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

import os
import time

import tensorflow as tf
import horovod.tensorflow as hvd

from dllogger.autologging import log_hardware
from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger import tags


class ProfilerHook(tf.train.SessionRunHook):

    def __init__(self, out_dir, global_batch_size, log_every=10, warmup_steps=20):
        LOGGER.set_model_name('UNet_TF')
        LOGGER.set_backends([
            dllg.JsonBackend(log_file=os.path.join(out_dir, 'dlloger_out.json'),
                             logging_scope=dllg.Scope.TRAIN_ITER, iteration_interval=1),
            dllg.StdOutBackend(log_file=None,
                               logging_scope=dllg.Scope.TRAIN_ITER, iteration_interval=log_every)

        ])

        self._perf = dllg.AverageMeter()

        LOGGER.register_metric('loss', meter=dllg.AverageMeter(), metric_scope=dllg.Scope.TRAIN_ITER)
        LOGGER.register_metric('dice_loss', meter=dllg.AverageMeter(), metric_scope=dllg.Scope.TRAIN_ITER)
        LOGGER.register_metric('total_loss', meter=dllg.AverageMeter(), metric_scope=dllg.Scope.TRAIN_ITER)

        self._warmup_steps = warmup_steps
        self._global_batch_size = global_batch_size
        self._current_step = 0

    def before_run(self, run_context):
        LOGGER.iteration_start()
        run_args = tf.train.SessionRunArgs(
            fetches=[
                'UNet/cross_loss_ref:0',
                'UNet/dice_loss_ref:0',
                'UNet/total_loss_ref:0']
        )
        self._t0 = time.time()

        return run_args

    def after_run(self,
                  run_context,
                  run_values):
        cross_loss, dice_loss, total_loss = run_values.results

        batch_time = time.time() - self._t0
        ips = self._global_batch_size / batch_time
        ips *= hvd.size()

        if self._current_step >= self._warmup_steps:
            LOGGER.log("iteration", int(self._current_step))
            LOGGER.log("loss", float(cross_loss))
            LOGGER.log("dice_loss", float(dice_loss))
            LOGGER.log("total_loss", float(total_loss))
            self._perf.record(ips)
            LOGGER.iteration_stop()

        self._current_step += 1

    def begin(self):
        log_hardware(LOGGER)
        LOGGER.log(tags.RUN_INIT)

    def end(self, session):
        LOGGER.log(tags.RUN_FINAL)
        LOGGER.finish()
        LOGGER.log("average_images_per_second", self._perf.get_value())
