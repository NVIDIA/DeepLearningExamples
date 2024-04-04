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

import atexit
import os
import time
from collections import OrderedDict
from functools import partial
from queue import Queue
from threading import Thread
from typing import Callable

import wandb
import mlflow
from distributed_utils import is_parallel
from dllogger import Backend
from mlflow.entities import Metric, Param
from torch.utils.tensorboard import SummaryWriter


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
        self.metrics = OrderedDict(agg_dict)
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
    def __init__(self, verbosity, log_dir='.'):
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

    def log_figure(self, fig, name, step):
        if not isinstance(step, int):
            return
        self.summary_writer.add_figure(name, fig, global_step=step)

    def flush(self):
        self.summary_writer.flush()

class AsyncCaller:
    STOP_MARK = "__STOP"

    def __init__(self) -> None:
        self._q = Queue()
        self._stop = False
        self._t = Thread(target=self.run, daemon=True)
        self._t.start()

    def close(self):
        self._q.put(self.STOP_MARK)

    def run(self):
        while True:
            data = self._q.get()
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        self._q.put(partial(func, *args, **kwargs))

    def wait(self, close=True):
        if close:
            self.close()
        self._t.join()

    @staticmethod
    def async_dec(ac_attr):
        def decorator_func(func):
            def wrapper(self, *args, **kwargs):
                if isinstance(getattr(self, ac_attr, None), Callable):
                    return getattr(self, ac_attr)(func, self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator_func


class WandBBackend(Backend):
    def __init__(self, verbosity):
        super().__init__(verbosity=verbosity)
        wandb.init()

    @property
    def log_level(self):
        return self._log_level

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass

    def log(self, timestamp, elapsedtime, step, data):
        close = step == [] or step == ()
        if step == 'PARAMETER':
            wandb.config.update(data)
        if not isinstance(step, int):
            step = None
        wandb.log(data={k: v for k,v in data.items() if isinstance(v, (float, int))}, step=step)
        if close:
            exit_code = 1 if not data else 0
            wandb.finish(exit_code=exit_code, quiet=True)
    
    def log_figure(self, fig, name, step):
        if not isinstance(step, int):
            return
        wandb.log({name: fig}, step=step)

    def flush(self):
        pass


class MLflowBackend(Backend):
    def __init__(self, uri, experiment_name, verbosity):
        super().__init__(verbosity=verbosity)
        assert not uri.startswith(
            "http") or experiment_name, "When specifying remote tracking server, experiment name is mandatory"

        self.client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        if experiment_name:
            exp = self.client.get_experiment_by_name(experiment_name)
            if exp is None:
                if is_parallel():
                    raise NotImplementedError("For parallel jobs create experiment beforehand")
                exp_id = self.client.create_experiment(experiment_name)
            else:
                exp_id = exp.experiment_id
        else:
            exp_id = '0'

        self.run = self.client.create_run(exp_id)

        self.async_caller = AsyncCaller()
        self.buffer = {'metrics': [], 'params': [], 'tags': []}

    def close(self):
        self.async_caller.close()

    @property
    def log_level(self):
        return self._log_level

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass

    def log(self, timestamp, elapsedtime, step, data):
        timestamp = int(timestamp.timestamp() * 1000)
        if step == 'PARAMETER':
            for k, v in data.items():
                self.buffer['params'].append(Param(k, str(v)))
        elif isinstance(step, int):
            for k, v in data.items():
                self.buffer['metrics'].append(Metric(k, v, timestamp, step))
        elif step == []:
            for k, v in data.items():
                self.buffer['metrics'].append(Metric(k, v, timestamp, 0))
            self.client.set_terminated(self.run.info.run_id)
            self.flush()
            self.async_caller.wait(close=True)

    @AsyncCaller.async_dec(ac_attr="async_caller")
    def flush(self):
        for b in self._batched_buffer():
            self.client.log_batch(self.run.info.run_id, **b)
        for k in self.buffer.keys():
            self.buffer[k] = []

    def _batched_buffer(self):
        while sum(len(v) for v in self.buffer.values()) > 0:
            batch = {}
            capacity = 1000
            for k, v in self.buffer.items():
                _v = v[:capacity]
                batch[k] = _v
                self.buffer[k] = v[capacity:]
                capacity -= len(_v)
            yield batch
