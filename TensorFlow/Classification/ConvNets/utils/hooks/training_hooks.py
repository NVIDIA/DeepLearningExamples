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
import tensorflow as tf

import dllogger
import signal

__all__ = ['TrainingLoggingHook', 'TrainingPartitionHook']


class MeanAccumulator:

    def __init__(self):
        self.sum = 0
        self.count = 0

    def consume(self, value):
        self.sum += value
        self.count += 1

    def value(self):
        if self.count:
            return self.sum / self.count
        else:
            return 0


class TrainingLoggingHook(tf.train.SessionRunHook):

    def __init__(
        self, global_batch_size, num_steps, num_samples, num_epochs, steps_per_epoch, warmup_steps=20, logging_steps=1
    ):
        self.global_batch_size = global_batch_size
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps

        self.current_step = 0
        self.current_epoch = 0
        self.t0 = None

        self.mean_throughput = MeanAccumulator()

    # Determines if its the last step of the epoch
    def _last_step_of_epoch(self, global_step):
        return (global_step + 1) // self.steps_per_epoch > (global_step // self.steps_per_epoch)

    def before_run(self, run_context):
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

        metrics = {
            "imgs_per_sec": ips,
            "cross_entropy": cross_entropy,
            "l2_loss": l2_loss,
            "total_loss": total_loss,
            "learning_rate": learning_rate
        }

        if self.current_step >= self.warmup_steps:
            self.mean_throughput.consume(metrics['imgs_per_sec'])

            if (self.current_step % self.logging_steps) == 0:
                metrics = {k: float(v) for k, v in metrics.items()}
                dllogger.log(data=metrics, step=(int(global_step // self.steps_per_epoch), int(global_step)))

        self.current_step += 1

        if self._last_step_of_epoch(global_step):
            metrics = {
                "cross_entropy": cross_entropy,
                "l2_loss": l2_loss,
                "total_loss": total_loss,
                "learning_rate": learning_rate
            }
            metrics = {k: float(v) for k, v in metrics.items()}
            dllogger.log(data=metrics, step=(int(global_step // self.steps_per_epoch), ))
            self.current_epoch += 1


class TrainingPartitionHook(tf.train.SessionRunHook):

    def __init__(self):
        super().__init__()
        self.should_exit = False

        signal.signal(signal.SIGUSR1, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def after_run(self, run_context, run_values):
        if self.should_exit:
            run_context.request_stop()

    def _signal_handler(self, signum, frame):
        print("Stop signal received")
        self.should_exit = True
