# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import datetime
import logging
import os
import shutil
import sys
import time

import dllogger
import torch.utils.collect_env

import utils


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, warmup=0, keep=False):
        self.reset()
        self.warmup = warmup
        self.keep = keep

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.iters = 0
        self.vals = []

    def update(self, val, n=1):
        self.iters += 1
        self.val = val

        if self.iters > self.warmup:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            if self.keep:
                self.vals.append(val)


def log_env_info():
    """
    Prints information about execution environment.
    """
    logging.info('Collecting environment information...')
    env_info = torch.utils.collect_env.get_pretty_env_info()
    logging.info(f'{env_info}')


def benchmark(test_perplexity=None, target_perplexity=None,
              test_throughput=None, target_throughput=None):
    def test(achieved, target, name, higher_better=True):
        passed = True
        if target is not None and achieved is not None:
            logging.info(f'{name} achieved: {achieved:.2f} '
                         f'target: {target:.2f}')
            if higher_better:
                result = (achieved >= target)
            else:
                result = (achieved <= target)

            if result:
                logging.info(f'{name} test passed')
            else:
                logging.info(f'{name} test failed')
                passed = False
        return passed

    passed = True
    passed &= test(test_perplexity, target_perplexity, 'Perplexity', False)
    passed &= test(test_throughput, target_throughput, 'Throughput')
    return passed


def setup_logging(log_all_ranks=True, filename=os.devnull, filemode='w'):
    """
    Configures logging.
    By default logs from all workers are printed to the console, entries are
    prefixed with "N: " where N is the rank of the worker. Logs printed to the
    console don't include timestaps.
    Full logs with timestamps are saved to the log_file file.
    """
    class RankFilter(logging.Filter):
        def __init__(self, rank, log_all_ranks):
            self.rank = rank
            self.log_all_ranks = log_all_ranks

        def filter(self, record):
            record.rank = self.rank
            if self.log_all_ranks:
                return True
            else:
                return (self.rank == 0)

    rank = utils.distributed.get_rank()
    rank_filter = RankFilter(rank, log_all_ranks)

    if log_all_ranks:
        logging_format = "%(asctime)s - %(levelname)s - %(rank)s - %(message)s"
    else:
        logging_format = "%(asctime)s - %(levelname)s - %(message)s"
        if rank != 0:
            filename = os.devnull

    logging.basicConfig(level=logging.DEBUG,
                        format=logging_format,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=filename,
                        filemode=filemode)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    if log_all_ranks:
        formatter = logging.Formatter('%(rank)s: %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)


def setup_dllogger(enabled=True, filename=os.devnull):
    rank = utils.distributed.get_rank()

    if enabled and rank == 0:
        backends = [
            dllogger.JSONStreamBackend(
                dllogger.Verbosity.VERBOSE,
                filename,
                ),
            ]
        dllogger.init(backends)
    else:
        dllogger.init([])


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        return

    os.makedirs(dir_path, exist_ok=True)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        os.makedirs(script_path, exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def build_work_dir_name(work_dir, dataset, append_dataset, append_time):
    if append_dataset:
        work_dir = '{}-{}'.format(work_dir, dataset)

    if append_time:
        now = int(time.time())
        now_max = utils.distributed.all_reduce_item(now, op='max')
        now_str = datetime.datetime.fromtimestamp(now_max).strftime('%Y%m%d-%H%M%S')

        work_dir = os.path.join(work_dir, now_str)
    return work_dir
