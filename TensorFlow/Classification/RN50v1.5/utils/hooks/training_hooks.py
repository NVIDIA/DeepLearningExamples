#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import dllogger.logger as dllg
from dllogger.logger import LOGGER

__all__ = ['TrainingLoggingHook']


class TrainingLoggingHook(tf.train.SessionRunHook):

    def __init__(
        self, log_file_path, global_batch_size, num_steps, num_samples, num_epochs, log_every=10, warmup_steps=20
    ):
        LOGGER.set_model_name('resnet')

        LOGGER.set_backends(
            [
                dllg.JsonBackend(
                    log_file=log_file_path, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=log_every
                ),
                dllg.StdOutBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=log_every)
            ]
        )

        #
        # Set-up the train_iter scope metrics
        LOGGER.register_metric("iteration", metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("imgs_per_sec", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("cross_entropy", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("l2_loss", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("total_loss", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("learning_rate", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)

        # Set up the eval-scope metrics
        LOGGER.register_metric("epoch", metric_scope=dllg.EPOCH_SCOPE)
        LOGGER.register_metric("final_cross_entropy", metric_scope=dllg.EPOCH_SCOPE)
        LOGGER.register_metric("final_l2_loss", metric_scope=dllg.EPOCH_SCOPE)
        LOGGER.register_metric("final_total_loss", metric_scope=dllg.EPOCH_SCOPE)
        LOGGER.register_metric("final_learning_rate", metric_scope=dllg.EPOCH_SCOPE)

        self.global_batch_size = global_batch_size
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps

        self.current_step = 0
        self.current_epoch = 0

    # Determines if its the last step of the epoch
    def _last_step_of_epoch(self):
        return (self.global_batch_size * (self.current_step + 1) > (self.current_epoch + 1) * self.num_samples)

    def before_run(self, run_context):
        LOGGER.iteration_start()
        run_args = tf.train.SessionRunArgs(
            fetches=[
                tf.train.get_global_step(), 'cross_entropy_loss_ref:0', 'l2_loss_ref:0', 'total_loss_ref:0',
                'learning_rate_ref:0'
            ]
        )
        self.t0 = time.time()

        return run_args

    def after_run(self, run_context, run_values):
        global_step, cross_entropy, l2_loss, total_loss, learning_rate = run_values.results
        batch_time = time.time() - self.t0
        ips = self.global_batch_size / batch_time

        LOGGER.log("iteration", int(self.current_step))
        LOGGER.log("imgs_per_sec", float(ips))
        LOGGER.log("cross_entropy", float(cross_entropy))
        LOGGER.log("l2_loss", float(l2_loss))
        LOGGER.log("total_loss", float(total_loss))
        LOGGER.log("learning_rate", float(learning_rate))
        LOGGER.iteration_stop()

        self.current_step += 1

        if self._last_step_of_epoch():
            LOGGER.epoch_start()
            LOGGER.log("epoch", int(self.current_epoch))
            LOGGER.log("final_cross_entropy", float(cross_entropy))
            LOGGER.log("final_l2_loss", float(l2_loss))
            LOGGER.log("final_total_loss", float(total_loss))
            LOGGER.log("final_learning_rate", float(learning_rate))
            LOGGER.epoch_stop()
            self.current_epoch += 1

    def end(self, session):
        LOGGER.finish()
