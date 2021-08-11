# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

import os
from contextlib import contextmanager

import torch
import math

import pynvml

pynvml.nvmlInit()


def init_distributed(cuda):
    """
    Initializes distributed backend.

    :param cuda: (bool) if True initializes nccl backend, if False initializes
        gloo backend
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = (world_size > 1)
    if distributed:
        backend = 'nccl' if cuda else 'gloo'
        torch.distributed.init_process_group(backend=backend,
                                             init_method='env://')
        assert torch.distributed.is_initialized()
    return distributed


def barrier():
    """
    Call torch.distributed.barrier() if distritubed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size

def get_device_count():
    """
    Gets total number of devices per node
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        nproc_per_node = torch.cuda.device_count()
    else:
        nproc_per_node = 1
    return nproc_per_node

def all_reduce_item(value, op='sum'):
    """
    All-reduces single scalar value if distributed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if op == 'sum' or op == 'mean':
            dop = torch.distributed.ReduceOp.SUM
        elif op == 'min':
            dop = torch.distributed.ReduceOp.MIN
        elif op == 'max':
            dop = torch.distributed.ReduceOp.MAX
        elif op == 'product':
            dop = torch.distributed.ReduceOp.PRODUCT
        else:
            raise RuntimeError('Unsupported reduce op')

        backend = torch.distributed.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device('cuda')
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device('cpu')
        else:
            raise RuntimeError('Unsupported distributed backend')

        tensor = torch.tensor(value, device=device)
        torch.distributed.all_reduce(tensor, dop)
        if op == 'mean':
            tensor /= get_world_size()
        ret = tensor.item()
    else:
        if torch.is_tensor(value):
            ret = value.item()
        else:
            ret = value
    return ret


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()

def systemGetDriverVersion():
    return pynvml.nvmlSystemGetDriverVersion()


def deviceGetCount():
    return pynvml.nvmlDeviceGetCount()


class device:
    # assume nvml returns list of 64 bit ints
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)

    def __init__(self, device_idx):
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def getName(self):
        return pynvml.nvmlDeviceGetName(self.handle)

    def getCpuAffinity(self):
        affinity_string = ''
        for j in pynvml.nvmlDeviceGetCpuAffinity(
            self.handle, device._nvml_affinity_elements
        ):
            # assume nvml returns list of 64 bit ints
            affinity_string = '{:064b}'.format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        return [i for i, e in enumerate(affinity_list) if e != 0]


def set_affinity(gpu_id=None):
    if gpu_id is None:
        gpu_id = int(os.getenv('LOCAL_RANK', 0))

    dev = device(gpu_id)
    os.sched_setaffinity(0, dev.getCpuAffinity())

    # list of ints representing the logical cores this process is now affinitied with
    return os.sched_getaffinity(0)