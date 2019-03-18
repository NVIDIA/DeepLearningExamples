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

__all__ = ['BenchmarkLoggingHook']


class BenchmarkLoggingHook(tf.train.SessionRunHook):

    def __init__(self, log_file_path, global_batch_size, log_every=10, warmup_steps=20):
        LOGGER.set_model_name('resnet')

        LOGGER.set_backends(
            [
                dllg.JsonBackend(
                    log_file=log_file_path, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=log_every
                ),
                dllg.JoCBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=log_every)
            ]
        )

        LOGGER.register_metric("iteration", metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("total_ips", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)

        self.warmup_steps = warmup_steps
        self.global_batch_size = global_batch_size
        self.current_step = 0

    def before_run(self, run_context):
        self.t0 = time.time()
        if self.current_step >= self.warmup_steps:
            LOGGER.iteration_start()

    def after_run(self, run_context, run_values):
        batch_time = time.time() - self.t0
        ips = self.global_batch_size / batch_time
        if self.current_step >= self.warmup_steps:
            LOGGER.log("iteration", int(self.current_step))
            LOGGER.log("total_ips", float(ips))
            LOGGER.iteration_stop()

        self.current_step += 1

    def end(self, session):
        LOGGER.finish()
