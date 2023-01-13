# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import os

import torch
import torch.distributed as dist


def print_once(*msg, local_rank=0):
    """Single stdout print with multiple processes."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*msg)
    elif int(os.environ.get('WORLD_SIZE', 1)) == 1:
        print(*msg)
    elif int(os.environ.get('RANK', 0)) == 0 and local_rank == 0:
        print(*msg)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def reduce_tensor(tensor, world_size, mean=True):
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mean:
        rt = rt.true_divide(world_size)
    return rt


def all_reduce_cpu_scalars(data, device=torch.device('cuda')):
    data_keys = list(data.keys())
    data_vals = list(data.values())
    tensor_vals = torch.tensor(data_vals, dtype=torch.double, device=device)
    dist.all_reduce(tensor_vals, op=dist.ReduceOp.SUM)
    data_vals = tensor_vals.cpu().numpy()
    return dict(zip(data_keys, data_vals))


def setup_distributed(local_rank):
    multi_gpu = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        print_once(f'Distributed training with {world_size} GPUs\n')
    else:
        world_size = 1

    return world_size
