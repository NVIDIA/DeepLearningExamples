# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import errno
import os
import time
from collections import defaultdict, deque

import dllogger
import torch
import torch.distributed as dist

from dlrm.utils.distributed import is_dist_avail_and_initialized


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def print(self, header=None):
        if not header:
            header = ''
        print_str = header
        for name, meter in self.meters.items():
            print_str += F"  {name}: {meter}"
        print(print_str)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def lr_step(optim, num_warmup_iter, current_step, base_lr, warmup_factor, decay_steps=0, decay_start_step=None):
    if decay_start_step is None:
        decay_start_step = num_warmup_iter

    new_lr = base_lr

    if decay_start_step < num_warmup_iter:
        raise ValueError('Learning rate warmup must finish before decay starts')

    if current_step <= num_warmup_iter:
        warmup_step = base_lr / (num_warmup_iter * (2 ** warmup_factor))
        new_lr = base_lr - (num_warmup_iter - current_step) * warmup_step

    steps_since_decay_start = current_step - decay_start_step
    if decay_steps != 0 and steps_since_decay_start > 0:
        already_decayed_steps = min(steps_since_decay_start, decay_steps)
        new_lr = base_lr * ((decay_steps - already_decayed_steps) / decay_steps) ** 2
        min_lr = 0.0000001
        new_lr = max(min_lr, new_lr)

    for param_group in optim.param_groups:
        param_group['lr'] = new_lr


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def init_logging(log_path):
    json_backend = dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                              filename=log_path)
    stdout_backend = dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)

    stdout_backend._metadata['best_auc'].update({'format': '0:.5f'})
    stdout_backend._metadata['best_epoch'].update({'format': '0:.2f'})
    stdout_backend._metadata['average_train_throughput'].update({'format': ':.2e'})
    stdout_backend._metadata['average_test_throughput'].update({'format': ':.2e'})

    dllogger.init(backends=[json_backend, stdout_backend])


class StepTimer():
    def __init__(self):
        self._previous = None
        self._new = None
        self.measured = None

    def click(self):
        self._previous = self._new
        self._new = time.time()

        if self._previous is not None:
            self.measured = self._new - self._previous


class LearningRateScheduler:
    """Polynomial learning rate decay for multiple optimizers and multiple param groups

    Args:
        optimizers (list): optimizers for which to apply the learning rate changes
        base_lrs (list): a nested list of base_lrs to use for each param_group of each optimizer
        warmup_steps (int): number of linear warmup steps to perform at the beginning of training
        warmup_factor (int)
        decay_steps (int): number of steps over which to apply poly LR decay from base_lr to 0
        decay_start_step (int): the optimization step at which to start decaying the learning rate
            if None will start the decay immediately after
        decay_power (float): polynomial learning rate decay power
        end_lr_factor (float): for each optimizer and param group:
            lr = max(current_lr_factor, end_lr_factor) * base_lr

    Example:
        lr_scheduler = LearningRateScheduler(optimizers=[optimizer], base_lrs=[[lr]],
                                             warmup_steps=100, warmup_factor=0,
                                             decay_start_step=1000, decay_steps=2000,
                                             decay_power=2, end_lr_factor=1e-6)

        for batch in data_loader:
            lr_scheduler.step()
            # foward, backward, weight update
    """
    def __init__(self, optimizers, base_lrs, warmup_steps, warmup_factor,
                 decay_steps, decay_start_step, decay_power=2, end_lr_factor=0):
        self.current_step = 0
        self.optimizers = optimizers
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.decay_steps = decay_steps
        self.decay_start_step = decay_start_step
        self.decay_power = decay_power
        self.end_lr_factor = end_lr_factor
        self.decay_end_step = self.decay_start_step + self.decay_steps

        if self.decay_start_step < self.warmup_steps:
            raise ValueError('Learning rate warmup must finish before decay starts')

    def _compute_lr_factor(self):
        lr_factor = 1

        if self.current_step <= self.warmup_steps:
            warmup_step = 1 / (self.warmup_steps * (2 ** self.warmup_factor))
            lr_factor = 1 - (self.warmup_steps - self.current_step) * warmup_step
        elif self.decay_start_step < self.current_step <= self.decay_end_step:
            lr_factor = ((self.decay_end_step - self.current_step) / self.decay_steps) ** self.decay_power
            lr_factor = max(lr_factor, self.end_lr_factor)
        elif self.current_step > self.decay_end_step:
            lr_factor = self.end_lr_factor

        return lr_factor

    def step(self):
        self.current_step += 1
        lr_factor = self._compute_lr_factor()

        for optim, base_lrs in zip(self.optimizers, self.base_lrs):
            for group_id, base_lr in enumerate(base_lrs):
                optim.param_groups[group_id]['lr'] = base_lr * lr_factor


def roc_auc_score(y_true, y_score):
    """ROC AUC score in PyTorch

    Args:
        y_true (Tensor):
        y_score (Tensor):
    """
    device = y_true.device
    y_true.squeeze_()
    y_score.squeeze_()
    if y_true.shape != y_score.shape:
        raise TypeError(F"Shape of y_true and y_score must match. Got {y_true.shape()} and {y_score.shape()}.")

    desc_score_indices = torch.argsort(y_score, descending=True)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = torch.nonzero(y_score[1:] - y_score[:-1]).squeeze()
    threshold_idxs = torch.cat([distinct_value_indices, torch.tensor([y_true.numel() - 1], device=device)])

    tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = torch.cat([torch.zeros(1, device=device), tps])
    fps = torch.cat([torch.zeros(1, device=device), fps])

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    area = torch.trapz(tpr, fpr).item()

    return area
