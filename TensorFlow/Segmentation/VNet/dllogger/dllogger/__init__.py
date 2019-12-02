# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from .logger import (
    Backend,
    Verbosity,
    Logger,
    default_step_format,
    default_metric_format,
    StdOutBackend,
    JSONStreamBackend,
)

__version__ = "0.1.0"


class DLLoggerNotInitialized(Exception):
    pass


class DLLLoggerAlreadyInitialized(Exception):
    pass


class NotInitializedObject(object):
    def __getattribute__(self, name):
        raise DLLoggerNotInitialized(
            "DLLogger not initialized. Initialize DLLogger with init(backends) function"
        )


GLOBAL_LOGGER = NotInitializedObject()


def log(step, data, verbosity=Verbosity.DEFAULT):
    GLOBAL_LOGGER.log(step, data, verbosity=verbosity)


def metadata(metric, metadata):
    GLOBAL_LOGGER.metadata(metric, metadata)


def flush():
    GLOBAL_LOGGER.flush()


def init(backends):
    global GLOBAL_LOGGER
    try:
        if isinstance(GLOBAL_LOGGER, Logger):
            raise DLLLoggerAlreadyInitialized()
    except DLLoggerNotInitialized:
        GLOBAL_LOGGER = Logger(backends)
