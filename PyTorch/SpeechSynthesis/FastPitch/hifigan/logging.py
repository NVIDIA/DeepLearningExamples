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

import time
from collections import OrderedDict
from copy import copy
from pathlib import Path

import dllogger
import numpy as np
import torch.distributed as dist
import torch
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from common import tb_dllogger
from common.tb_dllogger import (stdout_metric_format, stdout_step_format,
                                unique_log_fpath, TBLogger)


def init_logger(output_dir, log_file, ema_decay=0.0):

    local_rank = 0 if not dist.is_initialized() else dist.get_rank()

    print('logger init', local_rank)

    if local_rank == 0:
        Path(output_dir).mkdir(parents=False, exist_ok=True)
        log_fpath = log_file or Path(output_dir, 'nvlog.json')

        dllogger.init(backends=[
            JSONStreamBackend(Verbosity.DEFAULT, unique_log_fpath(log_fpath)),
            StdOutBackend(Verbosity.VERBOSE, step_format=stdout_step_format,
                          metric_format=stdout_metric_format)])

        init_train_metadata()
    else:
        dllogger.init(backends=[])

    tb_train = ['train']
    tb_val = ['val']
    tb_ema = [k + '_ema' for k in tb_val] if ema_decay > 0.0 else []

    tb_dllogger.tb_loggers = {
        s: TBLogger(enabled=(local_rank == 0), log_dir=output_dir, name=s)
        for s in tb_train + tb_val + tb_ema}


def init_train_metadata():

    dllogger.metadata("train_lrate_gen",
                      {"name": "g lr", "unit": None, "format": ":>3.2e"})
    dllogger.metadata("train_lrate_discrim",
                      {"name": "d lr", "unit": None, "format": ":>3.2e"})
    dllogger.metadata("train_avg_lrate_gen",
                      {"name": "avg g lr", "unit": None, "format": ":>3.2e"})
    dllogger.metadata("train_avg_lrate_discrim",
                      {"name": "avg d lr", "unit": None, "format": ":>3.2e"})

    for id_, pref in [('train', ''), ('train_avg', 'avg train '),
                      ('val', '  avg val '), ('val_ema', '  EMA val ')]:

        dllogger.metadata(f"{id_}_loss_gen",
                          {"name": f"{pref}g loss", "unit": None, "format": ":>6.3f"})
        dllogger.metadata(f"{id_}_loss_discrim",
                          {"name": f"{pref}d loss", "unit": None, "format": ":>6.3f"})
        dllogger.metadata(f"{id_}_loss_mel",
                          {"name": f"{pref}mel loss", "unit": None, "format": ":>6.3f"})

        dllogger.metadata(f"{id_}_frames/s",
                          {"name": None, "unit": "frames/s", "format": ":>8.2f"})
        dllogger.metadata(f"{id_}_took",
                          {"name": "took", "unit": "s", "format": ":>3.2f"})


def init_infer_metadata():
    raise NotImplementedError

    # modalities = [('latency', 's', ':>10.5f'), ('RTF', 'x', ':>10.2f'),
    #               ('frames/s', None, ':>10.2f'), ('samples/s', None, ':>10.2f'),
    #               ('letters/s', None, ':>10.2f')]

    # for perc in ['', 'avg', '90%', '95%', '99%']:
    #     for model in ['fastpitch', 'waveglow', '']:
    #         for mod, unit, format in modalities:

    #             name = f'{perc} {model} {mod}'.strip().replace('  ', ' ')

    #             dllogger.metadata(
    #                 name.replace(' ', '_'),
    #                 {'name': f'{name: <26}', 'unit': unit, 'format': format})


class defaultdict(OrderedDict):
    """A simple, ordered defaultdict."""

    def __init__(self, type_, *args, **kwargs):
        self.type_ = type_
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self:
            self.__setitem__(key, self.type_())
        return super().__getitem__(key)

    def __copy__(self):
        return defaultdict(self.type_, self)


class Metrics(dict):

    def __init__(self, scopes=['train', 'train_avg'],
                 dll_keys=['loss_gen', 'loss_discrim', 'loss_mel',
                           'frames/s', 'took', 'lrate_gen', 'lrate_discrim'],
                 benchmark_epochs=0):
        super().__init__()

        self.dll_keys = dll_keys
        self.metrics = {scope: defaultdict(float) for scope in scopes}
        self.metric_counts = {scope: defaultdict(int) for scope in scopes}
        self.start_time = {scope: None for scope in scopes}
        self.benchmark_epochs = benchmark_epochs
        if benchmark_epochs > 0:
            self.metrics['train_benchmark'] = defaultdict(list)

    def __setitem__(self, key, val):
        extract = lambda t: t.item() if type(t) is torch.Tensor else t

        if type(val) is dict:
            for k, v in val.items():
                super().__setitem__(k, extract(v))
        else:
            super().__setitem__(key, extract(val))

    def __getitem__(self, key):
        if key not in self:
            self.__setitem__(key, 0.0)
        return super().__getitem__(key)

    def start_accumulating(self, step, start_timer=True, scope='train'):
        del step  # unused
        self.clear()
        self.metrics[scope].clear()
        self.metric_counts[scope].clear()
        if start_timer:
            self.start_time[scope] = time.time()

    def accumulate(self, scopes=['train', 'train_avg']):
        for scope in scopes:
            for k, v in self.items():
                self.metrics[scope][k] += v
                self.metric_counts[scope][k] += 1

        self.clear()

    def finish_accumulating(self, stop_timer=True, scope='train'):

        metr = self.metrics[scope]
        counts = self.metric_counts[scope]

        for k, v in metr.items():
            metr[k] = v / counts[k]

        if stop_timer:
            took = time.time() - self.start_time[scope]
            if 'frames' in metr:
                metr['frames/s'] = metr.pop('frames') * counts['frames'] / took
            metr['took'] = took

    def start_iter(self, iter, start_timer=True):
        self.start_accumulating(iter, start_timer, 'train')

    def start_epoch(self, epoch, start_timer=True):
        self.start_accumulating(epoch, start_timer, 'train_avg')

    def start_val(self, start_timer=True):
        self.start_accumulating(None, start_timer, 'val')

    def finish_iter(self, stop_timer=True):
        self.finish_accumulating(stop_timer, 'train')

    def finish_epoch(self, stop_timer=True):
        self.finish_accumulating(stop_timer, 'train_avg')

        metr = self.metrics['train_benchmark']
        for k in ('took', 'frames/s', 'loss_gen', 'loss_discrim', 'loss_mel'):
            metr[k].append(self.metrics['train_avg'][k])

            if len(metr[k]) > self.benchmark_epochs:
                metr[k].pop(0)

    def finish_val(self, stop_timer=True):
        self.finish_accumulating(stop_timer, 'val')

    def get_metrics(self, scope='train', target='dll'):

        if scope == 'train_benchmark':
            metr = self.metrics[scope]
            ret = {'train_' + k: np.mean(v) for k, v in metr.items()}
            ret['benchmark_epochs_num'] = len(list(metr.values())[0])
            return ret

        ret = copy(self.metrics[scope])

        if scope == 'train':
            ret.update(self)

        if target == 'dll':
            ret = {f'{scope}_{k}': v
                   for k, v in ret.items() if k in self.dll_keys}

        elif target == 'tb':
            # Rename keys so they would group nicely inside TensorBoard

            def split_key(k):
                pos = k.rfind('_')
                return k[:pos] + '/' + k[pos+1:] if pos >= 0 else k

            ret = {split_key(k): v for k, v in ret.items()}

        return ret
