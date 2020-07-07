# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import math
import numpy as np

import dllogger as DLLogger

class EpochMeter:
    def __init__(self, name):
        self.name = name
        self.data = []

    def update(self, epoch, val):
        self.data.append((epoch, val))


class IterationMeter:
    def __init__(self, name):
        self.name = name
        self.data = []

    def update(self, epoch, iteration, val):
        self.data.append((epoch, iteration, val))


class IterationAverageMeter:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.n = 0
        self.sum = 0

    def update_iter(self, val):
        if math.isfinite(val): # sometimes loss === 'inf'
            self.n += 1
            self.sum += 0 if math.isinf(val) else val

    def update_epoch(self, epoch):
        self.data.append((epoch, self.sum / self.n))
        self.n = 0
        self.sum = 0


class Logger:
    def __init__(self, name, json_output=None, print_freq=20):
        self.name = name
        self.train_loss_logger = IterationAverageMeter("Training loss")
        self.train_epoch_time_logger = EpochMeter("Training 1 epoch time")
        self.val_acc_logger = EpochMeter("Validation accuracy")
        self.print_freq = print_freq

        backends = [ DLLogger.StdOutBackend(DLLogger.Verbosity.DEFAULT) ]
        if json_output:
            backends.append(DLLogger.JSONStreamBackend(DLLogger.Verbosity.VERBOSE, json_output))

        DLLogger.init(backends)

        self.epoch = 0
        self.train_iter = 0
        self.summary = {}

    def step(self):
        return (
            self.epoch,
            self.train_iter,
        )

    def log_params(self, data):
        DLLogger.log("PARAMETER", data)
        DLLogger.flush()

    def log(self, key, value):
        DLLogger.log(self.step(), { key: value })
        DLLogger.flush()

    def add_to_summary(self, data):
        for key, value in data.items():
            self.summary[key] = value

    def log_summary(self):
        DLLogger.log((), self.summary)

    def update_iter(self, epoch, iteration, loss):
        self.train_iter = iteration
        self.train_loss_logger.update_iter(loss)
        if iteration % self.print_freq == 0:
            self.log('loss', loss)

    def update_epoch(self, epoch, acc):
        self.epoch = epoch
        self.train_loss_logger.update_epoch(epoch)
        self.val_acc_logger.update(epoch, acc)

        data = { 'mAP': acc }
        self.add_to_summary(data)
        DLLogger.log((self.epoch,), data)

    def update_epoch_time(self, epoch, time):
        self.epoch = epoch
        self.train_epoch_time_logger.update(epoch, time)
        DLLogger.log((self.epoch,), { 'time': time })

    def print_results(self):
        return self.train_loss_logger.data, self.val_acc_logger.data, self.train_epoch_time_logger


class BenchmarkMeter:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.total_images = 0
        self.total_time = 0
        self.avr_images_per_second = 0

    def update(self, bs, time):
        self.total_images += bs
        self.total_time += time
        self.avr_images_per_second = self.total_images / self.total_time
        self.data.append(bs / time)


class BenchLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_per_ses = BenchmarkMeter(self.name)

    def update(self, bs, time):
        self.images_per_ses.update(bs, time)

    def print_result(self):
        total_bs = self.images_per_ses.total_images
        total_time = self.images_per_ses.total_time
        avr = self.images_per_ses.avr_images_per_second

        data = np.array(self.images_per_ses.data)
        med = np.median(data)

        DLLogger.log((), {
            'avg_img/sec': avr,
            'med_img/sec': np.median(data),
            'min_img/sec': np.min(data),
            'max_img/sec': np.max(data),
        })
        print("Done benchmarking. Total images: {}\ttotal time: {:.3f}\tAverage images/sec: {:.3f}\tMedian images/sec: {:.3f}".format(
            total_bs,
            total_time,
            avr,
            med
        ))
        return med
