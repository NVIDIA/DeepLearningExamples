# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import pathlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Callable, Optional

import dllogger
import torch.distributed as dist
import wandb
from dllogger import Verbosity

from se3_transformer.runtime.utils import rank_zero_only


class Logger(ABC):
    @rank_zero_only
    @abstractmethod
    def log_hyperparams(self, params):
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
            elif isinstance(val, pathlib.Path) or isinstance(val, Enum):
                return str(val)
            return val

        return {key: _sanitize(val) for key, val in params.items()}


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


class DLLogger(Logger):
    def __init__(self, save_dir: pathlib.Path, filename: str):
        super().__init__()
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            dllogger.init(
                backends=[dllogger.JSONStreamBackend(Verbosity.DEFAULT, str(save_dir / filename))])

    @rank_zero_only
    def log_hyperparams(self, params):
        params = self._sanitize_params(params)
        dllogger.log(step="PARAMETER", data=params)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if step is None:
            step = tuple()

        dllogger.log(step=step, data=metrics)


class WandbLogger(Logger):
    def __init__(
            self,
            name: str,
            save_dir: pathlib.Path,
            id: Optional[str] = None,
            project: Optional[str] = None
    ):
        super().__init__()
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.experiment = wandb.init(name=name,
                                         project=project,
                                         id=id,
                                         dir=str(save_dir),
                                         resume='allow',
                                         anonymous='must')

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        params = self._sanitize_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is not None:
            self.experiment.log({**metrics, 'epoch': step})
        else:
            self.experiment.log(metrics)
