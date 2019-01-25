import logging.config
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.collect_env


def generate_seeds(rng, size):
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds


def setup_seeds(master_seed, epochs, device):
    if master_seed is None:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**32 - 1)
        if get_rank() == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            logging.info(f'Using random master seed: {master_seed}')
    else:
        # master seed was specified from command line
        logging.info(f'Using master seed from command line: {master_seed}')

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


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
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


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()


@contextmanager
def timer(name, ndigits=2, sync_gpu=True):
    if sync_gpu:
        torch.cuda.synchronize()
    start = time.time()
    yield
    if sync_gpu:
        torch.cuda.synchronize()
    stop = time.time()
    elapsed = round(stop - start, ndigits)
    logging.info(f'TIMER {name} {elapsed}')


def setup_logging(log_file=os.devnull):
    """
    Configures logging.
    By default logs from all workers are printed to the console, entries are
    prefixed with "N: " where N is the rank of the worker. Logs printed to the
    console don't include timestaps.
    Full logs with timestamps are saved to the log_file file.
    """
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            self.rank = rank

        def filter(self, record):
            record.rank = self.rank
            return True

    rank = get_rank()
    rank_filter = RankFilter(rank)

    logging_format = "%(asctime)s - %(levelname)s - %(rank)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=logging_format,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(rank)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)


def set_device(args):
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = (world_size > 1)
    if distributed:
        backend = 'nccl' if args.cuda else 'gloo'
        dist.init_process_group(backend=backend,
                                init_method='env://')
        assert dist.is_initialized()
    return distributed


def log_env_info():
    logging.info('Collecting environment information...')
    env_info = torch.utils.collect_env.get_pretty_env_info()
    logging.info(f'{env_info}')


def pad_vocabulary(math):
    if math == 'fp16':
        pad_vocab = 8
    elif math == 'fp32':
        pad_vocab = 1
    return pad_vocab


def benchmark(test_acc, target_acc, test_perf, target_perf):
    def test(achieved, target, name):
        passed = True
        if target is not None and achieved is not None:
            logging.info(f'{name} achieved: {achieved:.2f} '
                         f'target: {target:.2f}')
            if achieved >= target:
                logging.info(f'{name} test passed')
            else:
                logging.info(f'{name} test failed')
                passed = False
        return passed

    passed = True
    passed &= test(test_acc, target_acc, 'Accuracy')
    passed &= test(test_perf, target_perf, 'Performance')
    return passed

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, skip_first=True):
        self.reset()
        self.skip = skip_first

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val

        if self.skip:
            self.skip = False
        else:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def reduce(self, op):
        """
        Reduces average value over all workers.

        :param op: 'sum' or 'mean', reduction operator
        """
        if op not in ('sum', 'mean'):
            raise NotImplementedError

        distributed = (get_world_size() > 1)
        if distributed:
            # Backward/forward compatibility around
            # https://github.com/pytorch/pytorch/commit/540ef9b1fc5506369a48491af8a285a686689b36 and
            # https://github.com/pytorch/pytorch/commit/044d00516ccd6572c0d6ab6d54587155b02a3b86
            # To accomodate change in Pytorch's distributed API
            if hasattr(dist, "get_backend"):
                _backend = dist.get_backend()
                if hasattr(dist, "DistBackend"):
                    backend_enum_holder = dist.DistBackend
                else:
                    backend_enum_holder = dist.Backend
            else:
                _backend = dist._backend
                backend_enum_holder = dist.dist_backend

            cuda = _backend == backend_enum_holder.NCCL

            if cuda:
                avg = torch.cuda.FloatTensor([self.avg])
                _sum = torch.cuda.FloatTensor([self.sum])
            else:
                avg = torch.FloatTensor([self.avg])
                _sum = torch.FloatTensor([self.sum])

            _reduce_op = dist.reduce_op if hasattr(dist, "reduce_op") else dist.ReduceOp
            dist.all_reduce(avg, op=_reduce_op.SUM)
            dist.all_reduce(_sum, op=_reduce_op.SUM)
            self.avg = avg.item()
            self.sum = _sum.item()

            if op == 'mean':
                self.avg /= get_world_size()
                self.sum /= get_world_size()


def debug_tensor(tensor, name):
    """
    Simple utility which helps with debugging.
    Takes a tensor and outputs: min, max, avg, std, number of NaNs, number of
    INFs.

    :param tensor: torch tensor
    :param name: name of the tensor (only for logging)
    """
    logging.info(name)
    tensor = tensor.detach().float().cpu().numpy()
    logging.info(f'MIN: {tensor.min()} MAX: {tensor.max()} '
                 f'AVG: {tensor.mean()} STD: {tensor.std()} '
                 f'NAN: {np.isnan(tensor).sum()} INF: {np.isinf(tensor).sum()}')
