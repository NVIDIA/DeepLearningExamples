# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import time
import json
import logging
import os
import inspect
import sys
import re
from contextlib import contextmanager
import functools
from collections import OrderedDict

NVLOGGER_NAME = 'nv_dl_logger'
NVLOGGER_VERSION = '0.2.3'
NVLOGGER_TOKEN = ':::NVLOG'

MLPERF_NAME = 'mlperf_logger'
MLPERF_VERSION = '0.5.0'
MLPERF_TOKEN = ':::MLP'

DEFAULT_JSON_FILENAME = 'nvlog.json'

RUN_SCOPE = 0
EPOCH_SCOPE = 1
TRAIN_ITER_SCOPE = 2

_data = OrderedDict([
    ('model', None),
    ('epoch', -1),
    ('iteration', -1),
    ('total_iteration', -1),
    ('metrics', OrderedDict()),
    ('timed_blocks', OrderedDict()),
    ('current_scope', RUN_SCOPE)
    ])

def get_caller(root_dir=None):
    stack_files = [s[1].split('/')[-1] for s in inspect.stack()]
    stack_index = 0
    while stack_index < len(stack_files) and stack_files[stack_index] != 'logger.py':
        stack_index += 1
    while (stack_index < len(stack_files) and 
            stack_files[stack_index] in ['logger.py', 'autologging.py', 'contextlib.py']):
        stack_index += 1

    caller = inspect.stack()[stack_index]

    return "%s:%d" % (stack_files[stack_index], caller[2])

class StandardMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def record(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_last(self):
        return self.value

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.value = 0
        self.last = 0

    def record(self, value, n = 1):
        self.last = value
        self.count += n
        self.value += value * n

    def get_value(self):
        return self.value / self.count

    def get_last(self):
        return self.last

class JsonBackend(object):

    def __init__(self, log_file=DEFAULT_JSON_FILENAME, logging_scope=TRAIN_ITER_SCOPE,
            iteration_interval=1):
        self.log_file = log_file
        self.logging_scope = logging_scope
        self.iteration_interval = iteration_interval

        self.json_log = OrderedDict([
            ('run', OrderedDict()),
            ('epoch', OrderedDict()),
            ('iter', OrderedDict()),
            ('event', OrderedDict()),
            ])
        
        self.json_log['epoch']['x'] = []
        if self.logging_scope == TRAIN_ITER_SCOPE:
            self.json_log['iter']['x'] = [[]]

    def register_metric(self, key, metric_scope):
        if (metric_scope == TRAIN_ITER_SCOPE and 
                self.logging_scope == TRAIN_ITER_SCOPE):
            if not key in self.json_log['iter'].keys():
                self.json_log['iter'][key] = [[]]
        if metric_scope == EPOCH_SCOPE:
            if not key in self.json_log['epoch'].keys():
                self.json_log['epoch'][key] = []

    def log(self, key, value):
        if _data['current_scope'] == RUN_SCOPE:
            self.json_log['run'][key] = value
        elif _data['current_scope'] == EPOCH_SCOPE: 
            pass
        elif _data['current_scope'] == TRAIN_ITER_SCOPE:
            pass
        else:
            raise ValueError('log function for scope "', _data['current_scope'], 
                    '" not implemented')
    
    def log_event(self, key, value):
        if not key in self.json_log['event'].keys():
            self.json_log['event'][key] = []
        entry = OrderedDict()
        entry['epoch'] = _data['epoch']
        entry['iter'] = _data['iteration']
        entry['timestamp'] = time.time()
        if value:
            entry['value'] = value
        self.json_log['event'][key].append(entry)

    def log_iteration_summary(self):
        if (self.logging_scope == TRAIN_ITER_SCOPE and 
                _data['total_iteration'] % self.iteration_interval == 0):
            for key, m in _data['metrics'].items():
                if m.metric_scope == TRAIN_ITER_SCOPE:
                    self.json_log['iter'][key][-1].append(m.get_last())

            # log x for iteration number
            self.json_log['iter']['x'][-1].append(_data['iteration'])


    def dump_json(self):
        if self.log_file is None:
            print(json.dumps(self.json_log, indent=4))
        else:
            with open(self.log_file, 'w') as f:
                json.dump(self.json_log, fp=f, indent=4)

    def log_epoch_summary(self):
        for key, m in _data['metrics'].items():
            if m.metric_scope == EPOCH_SCOPE:
                self.json_log['epoch'][key].append(m.get_value())
            elif (m.metric_scope == TRAIN_ITER_SCOPE and 
                    self.logging_scope == TRAIN_ITER_SCOPE):
                # create new sublists for each iter metric in the next epoch
                self.json_log['iter'][key].append([])
        
        # log x for epoch number
        self.json_log['epoch']['x'].append(_data['epoch'])

        # create new sublist for iter's x in the next epoch
        if self.logging_scope == TRAIN_ITER_SCOPE:
            self.json_log['iter']['x'].append([])

        self.dump_json()

    def timed_block_start(self, name):
        pass

    def timed_block_stop(self, name):
        pass

    def finish(self):
        self.dump_json()

class _ParentStdOutBackend(object):

    def __init__(self, name, token, version, log_file, logging_scope, iteration_interval):

        self.root_dir = None
        self.worker = [0]
        self.prefix = ''

        self.name = name
        self.token = token
        self.version = version
        self.log_file = log_file
        self.logging_scope = logging_scope
        self.iteration_interval = iteration_interval

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        if (self.log_file == None):
            self.stream_handler = logging.StreamHandler(stream=sys.stdout)
            self.stream_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(self.stream_handler)
        else:
            self.file_handler = logging.FileHandler(self.log_file, mode='w')
            self.file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(self.file_handler)

    def register_metric(self, key, meter=None, metric_scope=EPOCH_SCOPE):
        pass

    def log_epoch_summary(self):
        pass

    def log_iteration_summary(self):
        pass

    def log(self, key, value):
        if _data['current_scope'] > self.logging_scope:
            pass
        elif (_data['current_scope'] == TRAIN_ITER_SCOPE and 
                _data['total_iteration'] % self.iteration_interval != 0):
            pass
        else:
            self.log_stdout(key, value)

    def log_event(self, key, value):
        self.log_stdout(key, value)
        
    def log_stdout(self, key, value=None, forced=False):
        # TODO: worker 0 
        # only the 0-worker will log
        #if not forced and self.worker != 0:
        #    pass

        if value is None:
            msg = key
        else:
            str_json = json.dumps(value)
            msg = '{key}: {value}'.format(key=key, value=str_json)

        call_site = get_caller(root_dir=self.root_dir)
        now = time.time()

        message = '{prefix}{token}v{ver} {model} {secs:.9f} ({call_site}) {msg}'.format(
            prefix=self.prefix, token=self.token, ver=self.version, secs=now, 
            model=_data['model'],
            call_site=call_site, msg=msg)
        
        self.logger.debug(message)

    def timed_block_start(self, name):
        self.log_stdout(key=name + "_start")

    def timed_block_stop(self, name):
        self.log_stdout(key=name + "_stop")

    def finish(self):
        pass

class StdOutBackend(_ParentStdOutBackend):

    def __init__(self, log_file=None, logging_scope=EPOCH_SCOPE, iteration_interval=1):
        _ParentStdOutBackend.__init__(self, name=NVLOGGER_NAME, token=NVLOGGER_TOKEN, 
                version=NVLOGGER_VERSION, log_file=log_file, logging_scope=logging_scope, 
                iteration_interval=iteration_interval)
        
class MLPerfBackend(_ParentStdOutBackend):

    def __init__(self, log_file=None, logging_scope=TRAIN_ITER_SCOPE, iteration_interval=1):
        _ParentStdOutBackend.__init__(self, name=MLPERF_NAME, token=MLPERF_TOKEN, 
                version=MLPERF_VERSION, log_file=log_file, logging_scope=logging_scope, 
                iteration_interval=iteration_interval)

class JoCBackend(_ParentStdOutBackend):

    def __init__(self, log_file=None, logging_scope=EPOCH_SCOPE, iteration_interval=1):
        _ParentStdOutBackend.__init__(self, name=NVLOGGER_NAME, token=NVLOGGER_TOKEN, 
                version=NVLOGGER_VERSION, log_file=log_file, logging_scope=logging_scope, 
                iteration_interval=iteration_interval)
    
    def finish(self):
        for k in _data['metrics'].keys():
            if isinstance(_data['metrics'][k], AverageMeter):
                super(JoCBackend, self).log('Average '+ k, _data['metrics'][k].get_value())
        
class _Logger(object):
    def __init__(self):

        self.backends = [
                StdOutBackend(),
                JsonBackend()
                ]
   
    def set_model_name(self, name):
        _data['model'] = name


    def set_backends(self, backends):
        self.backends = backends
        
    def register_metric(self, key, meter=None, metric_scope=EPOCH_SCOPE):
        if meter == None:
            meter = StandardMeter()
        #TODO: move to argument of Meter?
        meter.metric_scope = metric_scope
        _data['metrics'][key] = meter
        for b in self.backends:
            b.register_metric(key, metric_scope)

    def log(self, key, value=None, forced=False):
        if _data['current_scope'] == TRAIN_ITER_SCOPE or _data['current_scope'] == EPOCH_SCOPE:
            if key in _data['metrics'].keys():
                if _data['metrics'][key].metric_scope == _data['current_scope']:
                    _data['metrics'][key].record(value)
        for b in self.backends:
            b.log(key, value)

    def log_event(self, key, value=None):
        for b in self.backends:
            b.log_event(key, value)
    
    def timed_block_start(self, name):
        if not name in _data['timed_blocks']:
            _data['timed_blocks'][name] = OrderedDict()
        _data['timed_blocks'][name]['start'] = time.time()
        for b in self.backends:
            b.timed_block_start(name)
    
    def timed_block_stop(self, name):
        if not name in _data['timed_blocks']:
            raise ValueError('timed_block_stop called before timed_block_start for ' + name)
        _data['timed_blocks'][name]['stop'] = time.time()
        delta = _data['timed_blocks'][name]['stop'] - _data['timed_blocks'][name]['start']
        self.log(name + '_time', delta)
        for b in self.backends:
            b.timed_block_stop(name)

    def iteration_start(self):
        _data['current_scope'] = TRAIN_ITER_SCOPE
        _data['iteration'] += 1
        _data['total_iteration'] += 1


    def iteration_stop(self):
        for b in self.backends:
            b.log_iteration_summary()
        _data['current_scope'] = EPOCH_SCOPE

    def epoch_start(self):
        _data['current_scope'] = EPOCH_SCOPE 
        _data['epoch'] += 1
        _data['iteration'] = -1

        for n, m in _data['metrics'].items():
            if m.metric_scope == TRAIN_ITER_SCOPE:
                m.reset()

    def epoch_stop(self):
        for b in self.backends:
            b.log_epoch_summary()
        _data['current_scope'] = RUN_SCOPE

    def finish(self):
        for b in self.backends:
            b.finish()

    def iteration_generator_wrapper(self, gen):
        for g in gen:
            self.iteration_start()
            yield g
            self.iteration_stop()

    def epoch_generator_wrapper(self, gen):
        for g in gen:
            self.epoch_start()
            yield g
            self.epoch_stop()

LOGGER = _Logger()

@contextmanager
def timed_block(prefix, value=None, logger=LOGGER, forced=False):
    """ This function helps with timed blocks
        ----
        Parameters:
        prefix - one of items from TIMED_BLOCKS; the action to be timed
        logger - NVLogger object
        forced - if True then the events are always logged (even if it should be skipped)
    """
    if logger is None:
        pass
    logger.timed_block_start(prefix)
    yield logger
    logger.timed_block_stop(prefix)

def timed_function(prefix, variable=None, forced=False):
    """ This decorator helps with timed functions
        ----
        Parameters:
        prefix - one of items from TIME_BLOCK; the action to be timed
        logger - NVLogger object
        forced - if True then the events are always logged (even if it should be skipped)
    """
    def timed_function_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = kwargs.get('logger', LOGGER)
            value = kwargs.get(variable, next(iter(args), None))
            with timed_block(prefix=prefix, logger=logger, value=value, forced=forced):
                    func(*args, **kwargs)
        return wrapper
    return timed_function_decorator

