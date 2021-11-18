# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import subprocess
import time
from collections import OrderedDict

import dllogger
from dllogger import Backend, JSONStreamBackend, Logger, StdOutBackend
from torch.utils.tensorboard import SummaryWriter

from distributed_utils import is_main_process


class AverageMeter:
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


class PerformanceMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.updated = False
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
        return time.time() - self.start


class AggregatorBackend(Backend):
    def __init__(self, verbosity, agg_dict):
        super().__init__(verbosity=verbosity)
        self.metrics = OrderedDict({k: v() for k, v in agg_dict.items()})
        self.metrics.flushed = True
        self.step = 0
        self.epoch = 0
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
        # This method allows us to reset performance metrics in case we want to
        # exclude couple first iterations from performance measurement
        for name in self.metrics.keys():
            self._reset_perf_meter(name)

    def log(self, timestamp, elapsedtime, step, data):
        self.step = step
        if self.step == []:
            self.metrics.flushed = True
        if "epoch" in data.keys():
            self.epoch = data["epoch"]
        for k, v in data.items():
            if k not in self.metrics.keys():
                continue
            self.metrics.flushed = False
            self.metrics[k].update(v)

    def flush(self):
        if self.metrics.flushed:
            return
        result_string = "Epoch {} | step {} |".format(self.epoch, self.step)
        for name, agg in self.metrics.items():
            if not agg.updated:
                continue
            if isinstance(agg, AverageMeter):
                _name = "avg " + name
            elif isinstance(agg, PerformanceMeter):
                _name = name + "/s"

            result_string += _name + " {:.3f} |".format(agg.value)
            agg.reset()

        result_string += "walltime {:.3f} |".format(time.time() - self.start_time)
        self.metrics.flushed = True
        print(result_string)


class TensorBoardBackend(Backend):
    def __init__(self, verbosity, log_dir):
        super().__init__(verbosity=verbosity)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, "TB_summary"), flush_secs=120, max_queue=200)
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


def empty_step_format(step):
    return ""


def empty_prefix_format(timestamp):
    return ""


def no_string_metric_format(metric, metadata, value):
    unit = metadata["unit"] if "unit" in metadata.keys() else ""
    format = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    if metric == "String":
        return "{} {}".format(format.format(value) if value is not None else value, unit)
    return "{} : {} {}".format(metric, format.format(value) if value is not None else value, unit)


def setup_logger(config):
    log_path = config.get("log_path", os.getcwd())
    if is_main_process():
        backends = [
            TensorBoardBackend(verbosity=dllogger.Verbosity.VERBOSE, log_dir=log_path),
            JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE, filename=os.path.join(log_path, "log.json")),
            AggregatorBackend(verbosity=dllogger.Verbosity.VERBOSE, agg_dict={"loss": AverageMeter}),
            StdOutBackend(
                verbosity=dllogger.Verbosity.DEFAULT,
                step_format=empty_step_format,
                metric_format=no_string_metric_format,
                prefix_format=empty_prefix_format,
            ),
        ]

        logger = Logger(backends=backends)
    else:
        logger = Logger(backends=[])
    container_setup_info = get_framework_env_vars()
    logger.log(step="PARAMETER", data=container_setup_info, verbosity=dllogger.Verbosity.DEFAULT)

    logger.metadata("loss", {"unit": "nat", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    logger.metadata("val_loss", {"unit": "nat", "GOAL": "MINIMIZE", "STAGE": "VAL"})
    return logger


def get_framework_env_vars():
    # TODO: it fails. Probably due to the fact that docker don't copy hidden directories
    # process = subprocess.Popen(
    #     ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    # )
    return {
        "NVIDIA_PYTORCH_VERSION": os.environ.get("NVIDIA_PYTORCH_VERSION"),
        "PYTORCH_VERSION": os.environ.get("PYTORCH_VERSION"),
        "CUBLAS_VERSION": os.environ.get("CUBLAS_VERSION"),
        "NCCL_VERSION": os.environ.get("NCCL_VERSION"),
        "CUDA_DRIVER_VERSION": os.environ.get("CUDA_DRIVER_VERSION"),
        "CUDNN_VERSION": os.environ.get("CUDNN_VERSION"),
        "CUDA_VERSION": os.environ.get("CUDA_VERSION"),
        "NVIDIA_PIPELINE_ID": os.environ.get("NVIDIA_PIPELINE_ID"),
        "NVIDIA_BUILD_ID": os.environ.get("NVIDIA_BUILD_ID"),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
