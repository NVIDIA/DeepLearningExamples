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

import pathlib
from abc import ABC, abstractmethod
from typing import Callable

import dllogger
from dllogger import Verbosity

from runtime.utils import rank_zero_only


class Logger(ABC):
    @rank_zero_only
    @abstractmethod
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    @abstractmethod
    def log_metadata(self, metric, metadata):
        pass

    @rank_zero_only
    @abstractmethod
    def log_metrics(self, metrics, step=None):
        pass

    @staticmethod
    def _sanitize_params(params):
        def _sanitize(val):
            if isinstance(val, Callable):
                try:
                    _val = val()
                    if isinstance(_val, Callable):
                        return val.__name__
                    return _val
                except Exception:
                    return getattr(val, "__name__", None)
            elif isinstance(val, pathlib.Path):
                return str(val)
            return val

        return {key: _sanitize(val) for key, val in params.items()}

    @rank_zero_only
    def flush(self):
        pass


class LoggerCollection(Logger):
    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def __getitem__(self, index):
        return [logger for logger in self.loggers][index]

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    @rank_zero_only
    def log_hyperparams(self, params):
        for logger in self.loggers:
            logger.log_hyperparams(params)

    @rank_zero_only
    def log_metadata(self, metric, metadata):
        for logger in self.loggers:
            logger.log_metadata(metric, metadata)

    @rank_zero_only
    def flush(self):
        for logger in self.loggers:
            logger.flush()


class DLLogger(Logger):
    def __init__(self, save_dir, filename, append, quiet):
        super().__init__()
        self._initialize_dllogger(save_dir, filename, append, quiet)

    @rank_zero_only
    def _initialize_dllogger(self, save_dir, filename, append, quiet):
        save_dir.mkdir(parents=True, exist_ok=True)
        backends = [
            dllogger.JSONStreamBackend(Verbosity.DEFAULT, str(save_dir / filename), append=append),
        ]
        if not quiet:
            backends.append(dllogger.StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: f"Step: {step} "))
        dllogger.init(backends=backends)

    @rank_zero_only
    def log_hyperparams(self, params):
        params = self._sanitize_params(params)
        dllogger.log(step="PARAMETER", data=params)

    @rank_zero_only
    def log_metadata(self, metric, metadata):
        dllogger.metadata(metric, metadata)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if step is None:
            step = tuple()
        dllogger.log(step=step, data=metrics)

    @rank_zero_only
    def flush(self):
        dllogger.flush()


def get_logger(args):
    loggers = []
    if args.use_dllogger:
        loggers.append(
            DLLogger(save_dir=args.results, filename=args.logname, append=args.resume_training, quiet=args.quiet)
        )
    return LoggerCollection(loggers)
