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

import atexit
import glob
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import dllogger
from common.utils import plot_spectrogram

tb_loggers = {}


class TBLogger:
    """
    xyz_dummies: stretch the screen with empty plots so the legend would
                 always fit for other plots
    """
    def __init__(self, enabled, log_dir, name, interval=1, dummies=True):
        self.enabled = enabled
        self.interval = interval
        self.cache = {}
        if self.enabled:
            self.summary_writer = SummaryWriter(
                log_dir=Path(log_dir, name), flush_secs=120, max_queue=200)
            atexit.register(self.summary_writer.close)
            if dummies:
                for key in ('_', '✕'):
                    self.summary_writer.add_scalar(key, 0.0, 1)

    def log(self, step, data):
        for k, v in data.items():
            self.log_value(step, k, v.item() if type(v) is torch.Tensor else v)

    def log_value(self, step, key, val, stat='mean'):
        if self.enabled:
            if key not in self.cache:
                self.cache[key] = []
            self.cache[key].append(val)
            if len(self.cache[key]) == self.interval:
                agg_val = getattr(np, stat)(self.cache[key])
                self.summary_writer.add_scalar(key, agg_val, step)
                del self.cache[key]

    def log_grads(self, step, model):
        if self.enabled:
            norms = [p.grad.norm().item() for p in model.parameters()
                     if p.grad is not None]
            for stat in ('max', 'min', 'mean'):
                self.log_value(step, f'grad_{stat}', getattr(np, stat)(norms),
                               stat=stat)

    def log_samples(self, step, sample_ind, audio, spec, rate):
        if self.enabled:
            log_prefix = 'gt/y' if step == 0 else 'generated/y_hat'

            self.summary_writer.add_audio(
                f'{log_prefix}_{sample_ind}', audio[0], step, rate)

            self.summary_writer.add_figure(
                f'{log_prefix}_spec_{sample_ind}',
                plot_spectrogram(spec[0].cpu().numpy()),
                step)


def unique_log_fpath(fpath):
    """Have a unique log filename for every separate run"""
    log_num = max([0] + [int(re.search("\.(\d+)", Path(f).suffix).group(1))
                         for f in glob.glob(f"{fpath}.*")])
    return f"{fpath}.{log_num + 1}"


def stdout_step_format(step):
    if isinstance(step, str):
        return step
    fields = []
    if len(step) > 0:
        fields.append("epoch {:>4}".format(step[0]))
    if len(step) > 1:
        fields.append("iter {:>3}".format(step[1]))
    if len(step) > 2:
        fields[-1] += "/{}".format(step[2])
    return " | ".join(fields)


def stdout_metric_format(metric, metadata, value):
    name = metadata.get("name", metric + " : ")
    unit = metadata.get("unit", None)
    format = f'{{{metadata.get("format", "")}}}'
    fields = [name, format.format(value) if value is not None else value, unit]
    fields = [f for f in fields if f is not None]
    return "| " + " ".join(fields)


def log(when, metrics={}, scope='train', flush_log=False, tb_iter=None):

    dllogger.log(when, data=metrics.get_metrics(scope, 'dll'))

    if tb_iter is not None:
        tb_loggers[scope].log(tb_iter, metrics.get_metrics(scope, 'tb'))

    if flush_log:
        flush()


def log_grads_tb(tb_total_steps, grads, tb_subset='train'):
    tb_loggers[tb_subset].log_grads(tb_total_steps, grads)


def log_samples_tb(tb_total_steps, sample_i, y, y_spec, rate, tb_subset='val',):
    tb_loggers[tb_subset].log_samples(tb_total_steps, sample_i, y, y_spec, rate)


def parameters(data, verbosity=0, tb_subset=None):
    for k, v in data.items():
        dllogger.log(step="PARAMETER", data={k: v}, verbosity=verbosity)

    if tb_subset is not None and tb_loggers[tb_subset].enabled:
        tb_data = {k: v for k, v in data.items()
                   if type(v) in (str, bool, int, float)}
        tb_loggers[tb_subset].summary_writer.add_hparams(tb_data, {})


def flush():
    dllogger.flush()
    for tbl in tb_loggers.values():
        if tbl.enabled:
            tbl.summary_writer.flush()
