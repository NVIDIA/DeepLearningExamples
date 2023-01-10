# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


import logging
import os

import torch
import torch.distributed as dist


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("not using a(ny) GPU(s)!")
    return device


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def reduce_tensor(tensor: torch.Tensor, num_gpus: int) -> torch.Tensor:
    if num_gpus > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        if rt.is_floating_point():
            rt = rt / num_gpus
        else:
            rt = rt // num_gpus
        return rt
    return tensor


def init_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0" # Needed for CUDA graphs
        dist.init_process_group(backend=backend, init_method="env://")
        assert dist.is_initialized()

    if get_rank() == 0:
        logging.info(f"Distributed initialized. World size: {world_size}")
    return distributed


def get_rank() -> int:
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank
