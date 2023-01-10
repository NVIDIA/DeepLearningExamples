# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


from abc import ABC, abstractmethod
import logging
import time

import dllogger
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
import numpy as np


LOGGING_LEVELS = dict(enumerate([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]))


def get_dllogger(args):
    backends = []
    if args.local_rank == 0:
        backends.append(StdOutBackend(Verbosity.VERBOSE))
        if args.log_path is not None:
            backends.append(JSONStreamBackend(Verbosity.VERBOSE, args.log_path, append=True))
    dllogger.init(backends=backends)
    return dllogger


def setup_logging(args):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:\t%(message)s', datefmt='%H:%M:%S', level=LOGGING_LEVELS[args.verbosity], force=True
    )
    return get_dllogger(args)


class BaseLogger(ABC):
    @abstractmethod
    def update(self, **kwargs) -> None:
        pass

    @abstractmethod
    def process_stats(self) -> dict:
        return {}

    @abstractmethod
    def reset(self) -> None:
        pass

    def summarize(self, step: tuple) -> None:
        stats = self.process_stats()
        if len(stats) == 0:
            logging.warn('Empty stats for logging, skipping')
            return
        self.logger.log(step=step, data=stats)
        self.logger.flush()


class PerformanceLogger(BaseLogger):
    def __init__(self, logger, batch_size: int, warmup_steps: int = 100, mode: str = 'train'):
        self.logger = logger
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self._step = 0
        self._timestamps = []
        self.mode = mode

    def update(self, **kwargs) -> None:
        self._step += 1
        if self._step >= self.warmup_steps:
            self._timestamps.append(time.time())
    
    def reset(self) -> None:
        self._step = 0
        self._timestamps = []

    def process_stats(self) -> dict:
        if len(self._timestamps) < 2:
            logging.warn('Cannot process performance stats - less than 2 measurements collected')
            return {}

        timestamps = np.asarray(self._timestamps)
        deltas = np.diff(timestamps)
        throughput = (self.batch_size / deltas).mean()
        stats = {
            f'throughput_{self.mode}': throughput,
            f'latency_{self.mode}_mean': deltas.mean(),
            f'total_time_{self.mode}': timestamps[-1] - timestamps[0],
        }
        for level in [90, 95, 99]:
            stats.update({f'latency_{self.mode}_{level}': np.percentile(deltas, level)})

        return stats


class MetricsLogger(BaseLogger):
    def __init__(self, logger, mode: str = 'train'):
        self.logger = logger
        self.mode = mode
        self._metrics_dict = {}

    def update(self, metrics: dict, **kwargs) -> None:
        for metrics_name, metric_val in metrics.items():
            if metrics_name not in self._metrics_dict:
                self._metrics_dict[metrics_name] = []
            self._metrics_dict[metrics_name].append(float(metric_val))
    
    def reset(self) -> None:
        self._metrics_dict = {}

    def process_stats(self) -> dict:
        stats = {}
        for metric_name, metric_val in self._metrics_dict.items():
            stats[metric_name] = np.mean(metric_val)
        return stats
