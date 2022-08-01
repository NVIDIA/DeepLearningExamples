# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import atexit
import time

from collections import OrderedDict
from threading import Thread
from queue import Queue
from functools import partial
from typing import Callable

from torch.utils.tensorboard import SummaryWriter
from dllogger import Backend

from distributed_utils import is_parallel

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.updated = False
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.updated = True
        if isinstance(value, (tuple, list)):
            val = value[0]
            n = value[1]
        else:
            val = value
            n = 1
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.avg


class PerformanceMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.updated = False
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.updated = True
        self.n += val

    @property
    def value(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return time.time() - self.start


class AggregatorBackend(Backend):
    def __init__(self, verbosity, agg_dict):
        super().__init__(verbosity=verbosity)
        self.metrics = OrderedDict({k: v() for k, v in agg_dict.items()})
        self.metrics.flushed = True
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()

    @property
    def log_level(self):
        return self._log_level

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass

    def _reset_perf_meter(self, name):
        for agg in self.metrics[name]:
            if isinstance(agg, PerformanceMeter):
                agg.reset()

    def reset_perf_meters(self):
        # This method allows us to reset performance metrics in case we want to
        # exclude couple first iterations from performance measurement
        for name in self.metrics.keys():
            self._reset_perf_meter(name)

    def log(self, timestamp, elapsedtime, step, data):
        self.step = step
        if self.step == []:
            self.metrics.flushed = True
        if "epoch" in data.keys():
            self.epoch = data["epoch"]
        for k, v in data.items():
            if k not in self.metrics.keys():
                continue
            self.metrics.flushed = False
            self.metrics[k].update(v)

    def flush(self):
        if self.metrics.flushed:
            return
        result_string = "Epoch {} | step {} |".format(self.epoch, self.step)
        for name, agg in self.metrics.items():
            if not agg.updated:
                continue
            if isinstance(agg, AverageMeter):
                _name = "avg " + name
            elif isinstance(agg, PerformanceMeter):
                _name = name + "/s"

            result_string += _name + " {:.3f} |".format(agg.value)
            agg.reset()

        result_string += "walltime {:.3f} |".format(time.time() - self.start_time)
        self.metrics.flushed = True
        print(result_string)


class TensorBoardBackend(Backend):
    def __init__(self, verbosity, log_dir='.'):
        super().__init__(verbosity=verbosity)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, "TB_summary"), flush_secs=120, max_queue=200)
        atexit.register(self.summary_writer.close)

    @property
    def log_level(self):
        return self._log_level

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass

    def log(self, timestamp, elapsedtime, step, data):
        if not isinstance(step, int):
            return
        for k, v in data.items():
            self.summary_writer.add_scalar(k, v, step)

    def flush(self):
        pass
