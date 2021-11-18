# SPDX-License-Identifier: Apache-2.0
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def load_checkpoint(load_ckpt_path):
    if load_ckpt_path:
        checkpoint = torch.load()
    else:
        checkpoint = None
    return checkpoint


def get_device(local_rank, device_name):
    if torch.cuda.is_available() and device_name == "cuda":
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def init_distributed(world_size):
    if dist.is_initialized():
        return True
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        assert dist.is_initialized()

    if get_rank() == 0:
        print("Distributed initialized. World size:", world_size)
    return distributed


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


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()


# XXX: Why do we even have 2 separate logging objects?
def log(to_log):
    if is_main_process():
        logging.info(to_log)
