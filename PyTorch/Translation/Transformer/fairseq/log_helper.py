import os
import atexit
import time
import itertools

from collections import OrderedDict

import dllogger
from dllogger import Backend, JSONStreamBackend
from tensorboardX import SummaryWriter
import torch


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.updated = False
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.updated = True
        if isinstance(value, (tuple, list)):
            val = value[0]
            n = value[1]
        else:
            val = value
            n = 1
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.avg


class PerformanceMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.updated = False
        torch.cuda.synchronize()
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.updated = True
        self.n += val

    @property
    def value(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        torch.cuda.synchronize()
        return time.time() - self.start


METRIC = {'average': AverageMeter, 'performance': PerformanceMeter}


class AggregatorBackend(Backend):
    def __init__(self, verbosity, agg_dict):
        super().__init__(verbosity=verbosity)
        agg_dict = OrderedDict({k: v if isinstance(v, (tuple, list)) else (v,) for k, v in agg_dict.items()})
        self.metrics = OrderedDict({k: [METRIC[x]() for x in v] for k, v in agg_dict.items()})
        self.metrics.flushed = True
        self.step = 0
        self.epoch = 0
        torch.cuda.synchronize()
        self.start_time = time.time()

    @property
    def log_level(self):
        return self._log_level

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass

    def _reset_perf_meter(self, name):
        for agg in self.metrics[name]:
            if isinstance(agg, PerformanceMeter):
                agg.reset()

    def reset_perf_meters(self):
        for name in self.metrics.keys():
            self._reset_perf_meter(name)

    def log(self, timestamp, elapsedtime, step, data):
        self.step = step
        if 'epoch' in data.keys():
            self.epoch = data['epoch']
        for k, v in data.items():
            if k not in self.metrics.keys():
                continue
            self.metrics.flushed = False
            for ag in self.metrics[k]:
                ag.update(v)

    def flush(self):
        if self.metrics.flushed:
            return
        result_string = 'Transformer | epoch {} | step {} |'.format(self.epoch, self.step)
        for name, aggregators in self.metrics.items():
            for agg in aggregators:
                if not agg.updated:
                    continue
                if isinstance(agg, AverageMeter):
                    _name = 'avg ' + name
                elif isinstance(agg, PerformanceMeter):
                    _name = name + '/s'

                result_string += _name + ' {:.3f} |'.format(agg.value)
                agg.reset()

        torch.cuda.synchronize()
        result_string += 'walltime {:.3f} |'.format(time.time() - self.start_time)
        self.metrics.flushed = True
        print(result_string)


class TensorBoardBackend(Backend):
    def __init__(self, verbosity, log_dir):
        super().__init__(verbosity=verbosity)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'TB_summary'),
                                            flush_secs=120,
                                            max_queue=200
                                            )
        atexit.register(self.summary_writer.close)

    @property
    def log_level(self):
        return self._log_level

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass

    def log(self, timestamp, elapsedtime, step, data):
        if not isinstance(step, int):
            return
        for k, v in data.items():
            self.summary_writer.add_scalar(k, v, step)

    def flush(self):
        pass


def setup_logger(args):
    aggregator_dict = OrderedDict([
        ('loss', 'average'),
        ('weighted_loss', 'average'),
        ('tokens', ('average', 'performance')),
        ('updates', 'performance'),
        ('gnorm', 'average')
    ])
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, args.stat_file)

    if os.path.exists(log_path):
        for i in itertools.count():
            s_fname = args.stat_file.split('.')
            fname = '.'.join(s_fname[:-1]) + f'_{i}.' + s_fname[-1] if len(s_fname) > 1 else args.stat_file + f'.{i}'
            log_path = os.path.join(args.save_dir, fname)
            if not os.path.exists(log_path):
                break

    if not args.distributed_world_size > 1 or args.distributed_rank == 0:
        dllogger.init(backends=[JSONStreamBackend(verbosity=1, filename=log_path),
                                AggregatorBackend(verbosity=0, agg_dict=aggregator_dict),
                                TensorBoardBackend(verbosity=1, log_dir=args.save_dir)])
    else:
        dllogger.init(backends=[])
    for k, v in vars(args).items():
        dllogger.log(step='PARAMETER', data={k: v}, verbosity=0)

    container_setup_info = get_framework_env_vars()
    dllogger.log(step='PARAMETER', data=container_setup_info, verbosity=0)

    dllogger.metadata('loss', {'unit': 'nat', 'GOAL': 'MINIMIZE', 'STAGE': 'TRAIN'})
    dllogger.metadata('val_loss', {'unit': 'nat', 'GOAL': 'MINIMIZE', 'STAGE': 'VAL'})
    dllogger.metadata('speed', {'unit': 'tokens/s', 'format': ':.3f', 'GOAL': 'MAXIMIZE', 'STAGE': 'TRAIN'})
    dllogger.metadata('accuracy', {'unit': 'bleu', 'format': ':.2f', 'GOAL': 'MAXIMIZE', 'STAGE': 'VAL'})


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


def reset_perf_meters():
    for backend in dllogger.GLOBAL_LOGGER.backends:
        if isinstance(backend, AggregatorBackend):
            backend.reset_perf_meters()
