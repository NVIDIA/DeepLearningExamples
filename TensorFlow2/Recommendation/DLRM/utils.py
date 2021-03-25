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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import time
import dllogger
import horovod.tensorflow as hvd
import json


def print_model_summary(model):
    variables_placement = {
        v.name: (v.device, v.shape.as_list(),
                 str(v.is_initialized()), str(v.dtype), str(v.trainable), str(v.synchronization)) for v in model.trainable_variables
    }
    print('============ VARIABLES PLACEMENT =====================')
    print(json.dumps(variables_placement, indent=4))
    print('============ VARIABLES PLACEMENT END =================')


def dist_print(*args, force=False, **kwargs):
    if hvd.rank() == 0 or force:
        print(*args, **kwargs)


def init_logging(log_path, FLAGS):
    json_backend = dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                              filename=log_path)
    stdout_backend = dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)

    stdout_backend._metadata['auc'].update({'format': '0:.5f'})
    stdout_backend._metadata['throughput'].update({'format': ':.2e'})
    stdout_backend._metadata['mean_step_time_ms'].update({'format': '0:.3f'})
    stdout_backend._metadata['mean_inference_throughput'].update({'format': ':.2e'})
    stdout_backend._metadata['mean_inference_latency'].update({'format': '0:.5f'})
    for percentile in [90, 95, 99]:
        stdout_backend._metadata[f'p{percentile}_inference_latency'].update({'format': '0:.5f'})

    dllogger.init(backends=[json_backend, stdout_backend])

    if hvd.rank() == 0:
        dllogger.log(data=FLAGS.flag_values_dict(), step='PARAMETER')
        print("Command line flags:")
        print(json.dumps(FLAGS.flag_values_dict(), indent=4))


class IterTimer:
    def __init__(self, train_batch_size, test_batch_size, optimizer, print_freq=50,
                 enabled=True, benchmark_warmup_steps=100):
        self.previous_tick = None
        self.train_idx = 0
        self.test_idx = 0
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.print_freq = print_freq
        self.optimizer = optimizer
        self.enabled = enabled
        self.training_steps_time = 0
        self.benchmark_warmup_steps = benchmark_warmup_steps
        self.steps_measured = 0

    def step_train(self, loss=None):
        if not self.enabled:
            return

        if self.train_idx < self.benchmark_warmup_steps:
            self.train_idx += 1
            return

        if self.train_idx % self.print_freq == 0 and self.train_idx > 0:
            if self.previous_tick is None:
                self.previous_tick = time.time()
                self.train_idx += 1
                return

            current_time = time.time()
            elapsed = current_time - self.previous_tick
            throughput = (self.train_batch_size * self.print_freq) / elapsed
            throughput_in_millions = throughput / 1e6
            step_time_ms = elapsed / self.print_freq * 1000
            lr = f'{self.optimizer.lr.numpy().item():.4f}'

            print(f'step={self.train_idx}, throughput={throughput_in_millions:.3f}M, step_time={step_time_ms:.3f} ms, learning_rate={lr}, loss={loss:.8f},')

            self.previous_tick = current_time
            self.training_steps_time += elapsed
            self.steps_measured += self.print_freq

        self.train_idx += 1

    def mean_train_time(self):
        return self.training_steps_time / self.steps_measured

    def step_test(self):
        if not self.enabled:
            return

        if self.previous_tick is None:
            self.previous_tick = time.time()
            self.test_idx += 1
            return

        if self.test_idx % self.print_freq == self.print_freq - 1:
            current_time = time.time()
            elapsed = current_time - self.previous_tick
            throughput = (self.test_batch_size * self.print_freq) / elapsed
            throughput_in_millions = throughput / 1e6
            step_time_ms = elapsed / self.print_freq * 1000

            print(f'validation_step={self.test_idx}, validation_throughput={throughput_in_millions:.3f}M, step_time={step_time_ms:.3f} ms')

            self.previous_tick = current_time
        self.test_idx += 1

