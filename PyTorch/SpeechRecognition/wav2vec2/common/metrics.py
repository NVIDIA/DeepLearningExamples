# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from collections import defaultdict
from copy import copy

import numpy as np
import torch

from common.utils import all_reduce_cpu_scalars, print_once


def __levenshtein(a, b):
    """Calculates the Levenshtein distance between two sequences."""

    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def word_error_rate(hypotheses, references):
    """Computes average Word Error Rate (WER) between two text lists."""

    scores = 0
    words = 0
    len_diff = len(references) - len(hypotheses)
    if len_diff > 0:
        raise ValueError("Uneqal number of hypthoses and references: "
                         "{0} and {1}".format(len(hypotheses), len(references)))
    elif len_diff < 0:
        hypotheses = hypotheses[:len_diff]

    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
    if words != 0:
        wer = 1.0*scores/words
    else:
        wer = float('inf')
    return wer, scores, words


class MetricsAggregator:
    def __init__(self, scopes=('train', 'train_avg'),
                 dllogger_keys=(),
                 benchmark_keys=(),
                 benchmark_epochs=0,
                 reduce_mean=(),
                 reduce_last=(),
                 group_tb_entries=False,
                 cuda=True):
        """
        Args:
            scopes: possible scopes of metrics accumulation
            dll_keys: metrics to log with dllogger
            benchmark_keys: metrics to log as benchmark metrics
            benchmark_epochs: num of last epochs to benchmark
        """
        super().__init__()

        self.dll_keys = dllogger_keys
        self.partials = defaultdict(float)
        self.partial_counts = defaultdict(int)
        self.accum_reductions = defaultdict(lambda: 'sum')
        self.accum_reductions.update({k: 'mean' for k in reduce_mean})
        self.accum_reductions.update({k: 'last' for k in reduce_last})
        self.metrics = {scope: defaultdict(float) for scope in scopes}
        self.metric_counts = {scope: defaultdict(int) for scope in scopes}
        self.start_time = {scope: None for scope in scopes}
        self.done_accumulating = {scope: True for scope in scopes}
        self.benchmark_epochs = benchmark_epochs
        self.metrics['train_benchmark'] = defaultdict(list)
        self.benchmark_keys = benchmark_keys
        self.scopes = scopes
        self.group_tb_entries = group_tb_entries
        self.cuda = cuda

    def log_scalar(self, key, val, accum_reduction=None):
        """Main primitive for logging partial metrics from single batch.

        NOTE: Assumption: `log_scalar` cannot be called with different
        `accum_reduction` for the same `key`. This results in undefined behavior

        Args:
            key: metric key
            val: metric value
            accum_reduction: defines how to accumulate given metric:
                - 'sum': sums metrics across grad acc and devices batches
                - 'mean': same as 'sum' but with averaging
                - 'last': overwrites previous accumulated values. Useful for
                    logging metric once in a grad acc batch, e.g. learning rate.
                If None, a default value is fetched from self.accum_reductions.
                If not None, overwrites defaults in self.accum_reductions
        """
        if accum_reduction is None:
            accum_reduction = self.accum_reductions[key]
        else:
            self.accum_reductions[key] = accum_reduction

        if accum_reduction == 'sum':
            self.partials[key] += val
            self.partial_counts[key] = 1
        elif accum_reduction == 'mean':
            self.partials[key] += val
            self.partial_counts[key] += 1
        elif accum_reduction == 'last':
            self.partials[key] = val  # overwrite accumulation
            self.partial_counts[key] = 1
        else:
            raise ValueError(accum_reduction)

    def log_scalars(self, scalars_dict, accum_reduction=None):
        """ Log whole dict of metrics at once """
        for k, v in scalars_dict.items():
            self.log_scalar(k, v, accum_reduction)

    def __setitem__(self, key, val):
        """ Convenience logging method. Use sparingly (see NOTE below).

        Uses 'last' aggregation and extracts tensors.

        Example:
        >>> metrics['lr'] = optim.param_groups[0]['lr']

        NOTE: `metrics['lr'] = ...` is very different
            from `metrics.partial['lr'] = ...`
        """
        extract = lambda t: t.item() if type(t) is torch.Tensor else t

        if type(val) is dict:
            for k, v in val.items():
                self.log_scalar(k, extract(v), 'last')
        else:
            self.log_scalar(key, extract(val), 'last')

    def accumulate(self, scopes=None):
        """ Accumulates partial metrics in metrics for given scopes.

        Defines boundaries of accum_reduction in `log_scalar` method.
        Intended to run after each gradient accumulation adjusted iteration.
        """
        scopes = scopes if scopes is not None else self.scopes
        for scope in scopes:
            for k, v in self.partials.items():
                self.metrics[scope][k] += v
                self.metric_counts[scope][k] += self.partial_counts.get(k, 1)

        self.partials.clear()
        self.partial_counts.clear()

    def all_reduce(self, world_size):
        """ Reduce metrics across devices.

        Currently assumes that all metrics are float scalars.

        After reducing, `log_scalar` method with accumulation other than 'last'
        shouldn't be called prior to calling `accumulate`.
        """
        if world_size == 1:
            return
        self.partials = defaultdict(float,
                                    all_reduce_cpu_scalars(self.partials))
        for k, v in self.partials.items():
            if self.accum_reductions[k] in ('mean', 'last'):
                self.partial_counts[k] *= (world_size - self.partials.get('ignore', 0))
                if self.partials.get('ignore', 0) > 0:
                    assert self.accum_reductions[k] == 'mean'
                    print_once(f'reducing with world size {world_size - self.partials.get("ignore", 0)}')

    def start_iter(self, iter):
        self._start_accumulating(iter, True, 'train')

    def start_epoch(self, epoch):
        if self.cuda:
            torch.cuda.synchronize()
        self._start_accumulating(epoch, True, 'train_avg')

    def start_val(self):
        if self.cuda:
            torch.cuda.synchronize()
        self._start_accumulating(None, True, 'val')

    def finish_iter(self):
        self._accumulate_time('train')

    def finish_logging_interval(self):
        self._finish_accumulating('train')

    def finish_epoch(self):
        if self.cuda:
            torch.cuda.synchronize()
        self._accumulate_time('train_avg')
        self._finish_accumulating('train_avg')

        metr = self.metrics['train_benchmark']
        for k in self.benchmark_keys:
            metr[k].append(self.metrics['train_avg'][k])

            if len(metr[k]) > self.benchmark_epochs:
                metr[k].pop(0)

    def finish_val(self, scope='val'):
        if self.cuda:
            torch.cuda.synchronize()
        self._accumulate_time(scope)
        self._finish_accumulating(scope)

    def get_metrics(self, scope='train', target='dll'):
        if scope == 'train_benchmark':
            metr = self.metrics[scope]
            ret = {'train_avg_' + k: np.mean(v) for k, v in metr.items()}
            ret['benchmark_epochs_num'] = len(list(metr.values())[0])
            return ret

        assert self.done_accumulating[scope]

        ret = copy(self.metrics[scope])

        if target == 'dll':
            ret = {f'{scope}_{k}': v
                   for k, v in ret.items() if k in self.dll_keys}

        elif target == 'tb' and self.group_tb_entries:
            # Rename keys so they would group nicely inside TensorBoard

            def split_key(k):
                pos = k.rfind('_')
                return k[:pos] + '/' + k[pos+1:] if pos >= 0 else k

            ret = {split_key(k): v for k, v in ret.items()}

        return ret

    def _start_accumulating(self, step, start_timer=True, scope='train'):
        del step  # unused
        assert not self.partials, 'metrics.accumulate call missed'
        assert not self.partial_counts, 'metrics.accumulate call missed'
        if self.done_accumulating[scope]:
            self.metrics[scope].clear()
            self.metric_counts[scope].clear()
        if start_timer:
            self.start_time[scope] = time.time()
        self.done_accumulating[scope] = False

    def _finish_accumulating(self, scope='train'):
        assert not self.done_accumulating[scope]
        metr = self.metrics[scope]
        counts = self.metric_counts[scope]

        for k, v in metr.items():
            metr[k] = v / counts[k]

        self.done_accumulating[scope] = True

    def _accumulate_time(self, scope='train'):
        assert not self.done_accumulating[scope]
        took = time.time() - self.start_time[scope]
        self.start_time[scope] = None
        self.metrics[scope]['took'] += took
        self.metric_counts[scope]['took'] = 1  # not +=
