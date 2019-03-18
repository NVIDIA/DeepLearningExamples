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


NVLOGGER_VERSION='0.1.0'
NVLOGGER_TOKEN= ':::NVLOG'
NVLOGGER_NAME="nv_dl_logger"
NVLOGGER_FILE_NAME="nv_dl_logger"

RUN_SCOPE = 0
EPOCH_SCOPE = 1
TRAIN_ITER_SCOPE = 2
EVAL_ITER_SCOPE = 3

LOGGING_SCOPE = {
    RUN_SCOPE,
    EPOCH_SCOPE,
    TRAIN_ITER_SCOPE,
    EVAL_ITER_SCOPE
}


def get_caller(stack_index=2, root_dir=None):
    caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

    # Trim the file names for readability.
    filename = caller.filename
    if root_dir is not None:
        filename = re.sub("^" + root_dir + "/", "", filename)
    return "%s:%d" % (filename, caller.lineno)


class NVLogger(object):
    __instance = None
    token = NVLOGGER_TOKEN
    version = NVLOGGER_VERSION
    stack_offset = 0
    extra_print = False
    model = "NN"
    root_dir = None
    worker = [0]
    prefix = ''
    log_file = None
    file_handler = None

    @staticmethod
    def get_instance():
        if NVLogger.__instance is None:
            NVLogger()

        return NVLogger.__instance

    def set_worker(self, worker):
        if worker is None:
            self.prefix = ''
            self.worker = [0]
        else:
            self.prefix = json.dumps(worker)
            self.worker = list(worker)

    def set_file(self, file_name=None):

        if file_name is None:
            self.log_file = os.getenv(NVLOGGER_FILE_NAME)
        else:
            self.log_file = file_name

        if self.log_file:
            self.file_handler = logging.FileHandler(self.log_file)
            self.file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(self.file_handler)
            self.stream_handler.setLevel(logging.INFO)
        else:
            self.stream_handler.setLevel(logging.DEBUG)

    def __init__(self):

        if NVLogger.__instance is None:
            NVLogger.__instance = self
        else:
            raise Exception("This class is a singleton!")

        self.logger = logging.getLogger(NVLOGGER_NAME)
        self.logger.setLevel(logging.DEBUG)

        self.stream_handler = logging.StreamHandler(stream=sys.stdout)
        self.stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.stream_handler)

    def print_vars(self, variables, forced=False, stack_offset=0):
        if isinstance(variables, dict):
            for v in variables.keys():
                self.log(key=v, value=variables[v], forced=forced, stack_offset=stack_offset+1)

    def print_vars2(self, key, variables, forced=False, stack_offset=0):
        if isinstance(variables, dict):
            self.log(key=key, value=variables, forced=forced, stack_offset=stack_offset+1)

    def log(self, key, value=None, forced=False, stack_offset=0):

        # only the 0-worker will log
        if not forced and self.worker != 0:
            pass

        if value is None:
            msg = key
        else:
            str_json = json.dumps(value)
            msg = '{key}: {value}'.format(key=key, value=str_json)

        call_site = get_caller(2 + self.stack_offset + stack_offset, root_dir=self.root_dir)
        now = time.time()

        message = msg
        if self.extra_print:
            print()

        self.logger.debug(message)


LOGGER = NVLogger.get_instance()

@contextmanager
def timed_block(prefix, value=None, logger=LOGGER, forced=False, stack_offset=2):
    """ This function helps with timed blocks
        ----
        Parameters:
        prefix - one of items from TIMED_BLOCKS; the action to be timed
        logger - NVLogger object
        forced - if True then the events are always logged (even if it should be skipped)
    """
    if logger is None:
        pass
    logger.log(key=prefix + "_start", value=value, forced=forced, stack_offset=stack_offset)
    yield logger
    logger.log(key=prefix + "_stop", forced=forced, stack_offset=stack_offset)


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
            with timed_block(prefix=prefix, logger=logger, value=value, forced=forced, stack_offset=3):
                    func(*args, **kwargs)
        return wrapper
    return timed_function_decorator
