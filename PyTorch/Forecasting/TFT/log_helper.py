# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import os
import subprocess
import sys
import itertools
import atexit

import dllogger
from dllogger import Backend, JSONStreamBackend, StdOutBackend

import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

class TensorBoardBackend(Backend):
    def __init__(self, verbosity, log_dir):
        super().__init__(verbosity=verbosity)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'TB_summary'),
                                            flush_secs=120,
                                            max_queue=200
                                            )
        self.hp_cache = None
        atexit.register(self.summary_writer.close)

    @property
    def log_level(self):
        return self._log_level

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass

    def log(self, timestamp, elapsedtime, step, data):
        if step == 'HPARAMS':
            parameters = {k: v for k, v in data.items() if not isinstance(v, (list, tuple))}
            #Unpack list and tuples
            for d in [{k+f'_{i}':v for i,v in enumerate(l)} for k,l in data.items() if isinstance(l, (list, tuple))]:
                parameters.update(d)
            #Remove custom classes
            parameters = {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}
            parameters.update({k:'None' for k, v in data.items() if v is None})
            self.hp_cache = parameters
        if step == ():
            if self.hp_cache is None:
                print('Warning: Cannot save HParameters. Please log HParameters with step=\'HPARAMS\'', file=sys.stderr)
                return
            self.summary_writer.add_hparams(self.hp_cache, data)
        if not isinstance(step, int):
            return
        for k, v in data.items():
            self.summary_writer.add_scalar(k, v, step)

    def flush(self):
        pass

def setup_logger(args):
    os.makedirs(args.results, exist_ok=True)
    log_path = os.path.join(args.results, args.log_file)

    if os.path.exists(log_path):
        for i in itertools.count():
            s_fname = args.log_file.split('.')
            fname = '.'.join(s_fname[:-1]) + f'_{i}.' + s_fname[-1] if len(s_fname) > 1 else args.stat_file + f'.{i}'
            log_path = os.path.join(args.results, fname)
            if not os.path.exists(log_path):
                break

    def metric_format(metric, metadata, value):
        return "{}: {}".format(metric, f'{value:.5f}' if isinstance(value, float) else value)
    def step_format(step):
        if step == ():
            return "Finished |"
        elif isinstance(step, int):
            return "Step {0: <5} |".format(step)
        return "Step {} |".format(step)


    if not dist.is_initialized() or not args.distributed_world_size > 1 or args.distributed_rank == 0:
        dllogger.init(backends=[JSONStreamBackend(verbosity=1, filename=log_path),
                                TensorBoardBackend(verbosity=1, log_dir=args.results),
                                StdOutBackend(verbosity=2, 
                                              step_format=step_format,
                                              prefix_format=lambda x: "")#,
                                              #metric_format=metric_format)
                                ])
    else:
        dllogger.init(backends=[])
    dllogger.log(step='PARAMETER', data=vars(args), verbosity=0)

    container_setup_info = {**get_framework_env_vars(), **get_system_info()}
    dllogger.log(step='ENVIRONMENT', data=container_setup_info, verbosity=0)

    dllogger.metadata('loss', {'GOAL': 'MINIMIZE', 'STAGE': 'TRAIN', 'format': ':5f', 'unit': None})
    dllogger.metadata('P10', {'GOAL': 'MINIMIZE', 'STAGE': 'TRAIN', 'format': ':5f', 'unit': None})
    dllogger.metadata('P50', {'GOAL': 'MINIMIZE', 'STAGE': 'TRAIN', 'format': ':5f', 'unit': None})
    dllogger.metadata('P90', {'GOAL': 'MINIMIZE', 'STAGE': 'TRAIN', 'format': ':5f', 'unit': None})
    dllogger.metadata('items/s', {'GOAL': 'MAXIMIZE', 'STAGE': 'TRAIN', 'format': ':1f', 'unit': 'items/s'})
    dllogger.metadata('val_loss', {'GOAL': 'MINIMIZE', 'STAGE': 'VAL', 'format':':5f', 'unit': None})
    dllogger.metadata('val_P10', {'GOAL': 'MINIMIZE', 'STAGE': 'VAL', 'format': ':5f', 'unit': None})
    dllogger.metadata('val_P50', {'GOAL': 'MINIMIZE', 'STAGE': 'VAL', 'format': ':5f', 'unit': None})
    dllogger.metadata('val_P90', {'GOAL': 'MINIMIZE', 'STAGE': 'VAL', 'format': ':5f', 'unit': None})
    dllogger.metadata('val_items/s', {'GOAL': 'MAXIMIZE', 'STAGE': 'VAL', 'format': ':1f', 'unit': 'items/s'})
    dllogger.metadata('test_P10', {'GOAL': 'MINIMIZE', 'STAGE': 'TEST', 'format': ':5f', 'unit': None})
    dllogger.metadata('test_P50', {'GOAL': 'MINIMIZE', 'STAGE': 'TEST', 'format': ':5f', 'unit': None})
    dllogger.metadata('test_P90', {'GOAL': 'MINIMIZE', 'STAGE': 'TEST', 'format': ':5f', 'unit': None})
    dllogger.metadata('sum', {'GOAL': 'MINIMIZE', 'STAGE': 'TEST', 'format': ':5f', 'unit': None})
    dllogger.metadata('throughput', {'GOAL': 'MAXIMIZE', 'STAGE': 'TEST', 'format': ':1f', 'unit': 'items/s'})
    dllogger.metadata('latency_avg', {'GOAL': 'MIMIMIZE', 'STAGE': 'TEST', 'format': ':5f', 'unit': 's'})
    dllogger.metadata('latency_p90', {'GOAL': 'MIMIMIZE', 'STAGE': 'TEST', 'format': ':5f', 'unit': 's'})
    dllogger.metadata('latency_p95', {'GOAL': 'MIMIMIZE', 'STAGE': 'TEST', 'format': ':5f', 'unit': 's'})
    dllogger.metadata('latency_p99', {'GOAL': 'MIMIMIZE', 'STAGE': 'TEST', 'format': ':5f', 'unit': 's'})
    dllogger.metadata('average_ips', {'GOAL': 'MAXIMIZE', 'STAGE': 'TEST', 'format': ':1f', 'unit': 'items/s'})


def get_framework_env_vars():
    return {
        'NVIDIA_PYTORCH_VERSION': os.environ.get('NVIDIA_PYTORCH_VERSION'),
        'PYTORCH_VERSION': os.environ.get('PYTORCH_VERSION'),
        'CUBLAS_VERSION': os.environ.get('CUBLAS_VERSION'),
        'NCCL_VERSION': os.environ.get('NCCL_VERSION'),
        'CUDA_DRIVER_VERSION': os.environ.get('CUDA_DRIVER_VERSION'),
        'CUDNN_VERSION': os.environ.get('CUDNN_VERSION'),
        'CUDA_VERSION': os.environ.get('CUDA_VERSION'),
        'NVIDIA_PIPELINE_ID': os.environ.get('NVIDIA_PIPELINE_ID'),
        'NVIDIA_BUILD_ID': os.environ.get('NVIDIA_BUILD_ID'),
        'NVIDIA_TF32_OVERRIDE': os.environ.get('NVIDIA_TF32_OVERRIDE'),
    }

def get_system_info():
    system_info = subprocess.run('nvidia-smi --query-gpu=gpu_name,memory.total,enforced.power.limit --format=csv'.split(), capture_output=True).stdout
    system_info = [i.decode('utf-8') for i in system_info.split(b'\n')]
    system_info = [x for x in system_info if x]
    return {'system_info': system_info}
