# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

# SPDX-License-Identifier: Apache-2.0
import os
import random

import torch
import torch.distributed as dist

from numba import cuda
import warnings
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from hydra.core.hydra_config import HydraConfig
from joblib.externals.loky.backend.context import get_context


def generate_seeds(rng, size):
    """
    Generate list of random seeds

    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2 ** 32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).

    :param seeds: list of seeds (integers)
    :param device: torch.device
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds

def setup_seeds(master_seed, epochs, device):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param master_seed: master RNG seed used to initialize other generators
    :param epochs: number of epochs
    :param device: torch.device (used for distributed.broadcast)
    """
    if master_seed == -1:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        if get_rank() == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            print(f"Using random master seed: {master_seed}")
    else:
        # master seed was specified from command line
        print(f"Using master seed from command line: {master_seed}")

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, get_world_size())

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, epochs)

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)
    return worker_seeds, shuffling_seeds


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def reduce_tensor(tensor, num_gpus, average=False):
    if num_gpus > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        if average:
            if rt.is_floating_point():
                rt = rt / num_gpus
            else:
                rt = rt // num_gpus
        return rt
    return tensor


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method="env://")
        assert dist.is_initialized()
        torch.cuda.set_device(local_rank)
        torch.cuda.synchronize()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def is_main_process():
    return get_rank() == 0


def init_parallel():
    if is_parallel():
        device_id = conf['device_id'] if 'device_id' in (conf := HydraConfig.get()) else conf.job.num % torch.cuda.device_count()
        torch.cuda.set_device(device_id)


def is_parallel():
    return HydraConfig.get().launcher.get('n_jobs', 0) > 1 or \
           HydraConfig.get().launcher.get('max_workers', 0) > 1 or \
           HydraConfig.get().launcher.get('processes', 0) > 1 or \
           HydraConfig.get().sweeper.get('n_jobs', 0) > 1


def get_mp_context():
    return get_context('loky')


def _pynvml_mem_size(kind="total", index=0):
    import pynvml

    pynvml.nvmlInit()
    size = None
    if kind == "free":
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).free)
    elif kind == "total":
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).total)
    else:
        raise ValueError("{0} not a supported option for device_mem_size.".format(kind))
    pynvml.nvmlShutdown()
    return size


def device_mem_size(kind="total"):

    if kind not in ["free", "total"]:
        raise ValueError("{0} not a supported option for device_mem_size.".format(kind))
    try:
        if kind == "free":
            return int(cuda.current_context().get_memory_info()[0])
        else:
            return int(cuda.current_context().get_memory_info()[1])
    except NotImplementedError:
        if kind == "free":
            # Not using NVML "free" memory, because it will not include RMM-managed memory
            warnings.warn("get_memory_info is not supported. Using total device memory from NVML.")
        size = _pynvml_mem_size(kind="total", index=0)
    return size


def get_rmm_size(size):
    return (size // 256) * 256


def calculate_frac(num_rows, num_feat, world_size):
    total_memory = world_size * device_mem_size(kind='total')
    mem_to_use = total_memory * 0.4
    num_rows_to_use = mem_to_use / (num_feat * 6)
    print(num_rows_to_use)
    frac = min(num_rows_to_use / num_rows, 1.0)
    return frac


def create_client(config):
    device_pool_frac = config.cluster.device_pool_frac # allocate 80% of total GPU memory on each GPU
    device_size = device_mem_size(kind="total")
    device_pool_size = int(device_pool_frac * device_size)
    dask_space = "/tmp/dask_space/"
    protocol = config.cluster.protocol
    visible_devices = [i for i in range(config.cluster.world_size)]
    if protocol == "ucx":
        cluster = LocalCUDACluster(
            protocol=protocol,
            CUDA_VISIBLE_DEVICES=visible_devices,
            rmm_pool_size=get_rmm_size(device_pool_size),
            local_directory=dask_space,
            device_memory_limit=None,
            enable_tcp_over_ucx=True,
            enable_nvlink=True)
    else:
        cluster = LocalCUDACluster(
            protocol=protocol,
            CUDA_VISIBLE_DEVICES=visible_devices,
            rmm_pool_size=get_rmm_size(device_pool_size),
            local_directory=dask_space,
            device_memory_limit=None,
        )
            
    client = Client(cluster)
    return client
