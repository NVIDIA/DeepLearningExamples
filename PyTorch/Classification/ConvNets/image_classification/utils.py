# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import os
import numpy as np
import torch
import shutil
import signal
import torch.distributed as dist


class Checkpointer:
    def __init__(self, last_filename, checkpoint_dir="./", keep_last_n=0):
        self.last_filename = last_filename
        self.checkpoints = []
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n

    def cleanup(self):
        to_delete = self.checkpoints[: -self.keep_last_n]
        self.checkpoints = self.checkpoints[-self.keep_last_n :]
        for f in to_delete:
            full_path = os.path.join(self.checkpoint_dir, f)
            os.remove(full_path)

    def get_full_path(self, filename):
        return os.path.join(self.checkpoint_dir, filename)

    def save_checkpoint(
        self,
        state,
        is_best,
        filename,
    ):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            assert False

        full_path = self.get_full_path(filename)

        print("SAVING {}".format(full_path))
        torch.save(state, full_path)
        self.checkpoints.append(filename)

        shutil.copyfile(
            full_path, self.get_full_path(self.last_filename)
        )

        if is_best:
            shutil.copyfile(
                full_path, self.get_full_path("model_best.pth.tar")
            )

        self.cleanup()


def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start

    return _timed_function


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    return rt


def first_n(n, generator):
    for i, d in zip(range(n), generator):
        yield d


class TimeoutHandler:
    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig
        self.device = torch.device("cuda")

    @property
    def interrupted(self):
        if not dist.is_initialized():
            return self._interrupted

        interrupted = torch.tensor(self._interrupted).int().to(self.device)
        dist.broadcast(interrupted, 0)
        interrupted = bool(interrupted.item())
        return interrupted

    def __enter__(self):
        self._interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def master_handler(signum, frame):
            self.release()
            self._interrupted = True
            print(f"Received SIGTERM")

        def ignoring_handler(signum, frame):
            self.release()
            print("Received SIGTERM, ignoring")

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            signal.signal(self.sig, master_handler)
        else:
            signal.signal(self.sig, ignoring_handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


def calc_ips(batch_size, time):
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    tbs = world_size * batch_size
    return tbs / time
