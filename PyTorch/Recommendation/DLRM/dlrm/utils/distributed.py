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

import os
from functools import reduce
from itertools import combinations_with_replacement
from typing import MutableSequence, Any, Sequence, List

import torch
import torch.distributed as dist


def setup_distributed_print(enable):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if enable or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_distributed() -> bool:
    return get_world_size() > 1


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(backend="nccl", use_gpu=True):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ and 'OMPI_COMM_WORLD_SIZE' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Not using distributed mode')
        return None, 1, None

    if use_gpu:
        torch.cuda.set_device(gpu)

    print('| distributed init (rank {})'.format(rank), flush=True)
    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank, init_method='env://')

    return rank, world_size, gpu


def get_gpu_batch_sizes(global_batch_size: int, num_gpus: int = 4, batch_std: int = 64, divisible_by: int = 64):
    batch_avg = global_batch_size // num_gpus
    start, end = batch_avg - batch_std, batch_avg + batch_std
    sizes_range = (x for x in range(start, end + 1) if x % divisible_by == 0)
    solutions = [
        sizes for sizes in combinations_with_replacement(sizes_range, num_gpus) if sum(sizes) == global_batch_size
    ]

    if not solutions:
        raise RuntimeError("Could not find GPU batch sizes for a given configuration. "
                           "Please adjust global batch size or number of used GPUs.")

    return max(solutions, key=lambda sizes: reduce(lambda x, y: x * y, sizes))


def distribute_to_buckets(elements: MutableSequence[Any], buckets: Sequence[List[Any]], start_bucket: int = 0):
    current_bucket = start_bucket % len(buckets)
    while elements:
        element = elements.pop()
        buckets[current_bucket].append(element)
        current_bucket = (current_bucket + 1) % len(buckets)
    return current_bucket


def get_criteo_device_mapping(num_gpus: int = 4, num_embeddings: int = 26, heavy_components=(0, 9, 19, 21, 20)):
    """Get device mappings for hybrid parallelism

    Bottom MLP running on device 0. 26 embeddings will be distributed across among all the devices. 0, 9, 19, 20, 21
    are the large ones, 20GB each.

    Args:
        num_gpus (int): Default 4.
        num_embeddings (int):
        heavy_components (tuple):

    Returns:
        device_mapping (dict):
    """
    bottom_mlp_index = -1
    heavy_components = list(heavy_components)
    regular_components = [x for x in range(num_embeddings) if x not in heavy_components]

    gpu_buckets = [[] for _ in range(num_gpus)]
    gpu_buckets[0].append(bottom_mlp_index)

    next_bucket = distribute_to_buckets(heavy_components, gpu_buckets, start_bucket=1)
    distribute_to_buckets(regular_components, gpu_buckets, start_bucket=next_bucket)

    vectors_per_gpu = [len(bucket) for bucket in gpu_buckets]

    gpu_buckets[0].pop(0)  # pop bottom mlp

    return {
        'bottom_mlp': 0,
        'embedding': gpu_buckets,
        'vectors_per_gpu': vectors_per_gpu,
    }
