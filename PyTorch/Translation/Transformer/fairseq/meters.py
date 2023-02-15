# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import time
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        torch.cuda.synchronize()
        self.start = time.time()
        self.n = 0
        self.last_update = time.time()

    def update(self, val=1):
        self.n += val
        torch.cuda.synchronize()
        self.last_update = time.time()

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        torch.cuda.synchronize()
        return self.init + (time.time() - self.start)

    @property
    def u_avg(self):
        return self.n / (self.last_update - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""
    def __init__(self):
        self.reset()
        self.intervals = []

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def stop(self, n=1):
        torch.cuda.synchronize()
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.intervals.append(delta)
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None
        self.intervals = []

    @property
    def avg(self):
        return self.sum / self.n

    def p(self, i):
        assert i <= 100
        idx = int(len(self.intervals) * i / 100)
        return sorted(self.intervals)[idx]
