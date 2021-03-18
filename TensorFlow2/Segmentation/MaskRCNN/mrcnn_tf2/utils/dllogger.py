# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import logging

from dllogger import Backend


class LoggingBackend(Backend):
    """ Simple DLLogger backend that uses python `logging` library. """

    def __init__(self, verbosity, logger_name='dllogger', level=logging.INFO):
        """ Creates backend for dllogger that uses `logging` library.

        Args:
            verbosity: DLLogger verbosity.
            logger_name: Name for `logging.Logger`.
            level: Logging level that will passed to `logging.Logger.log`.
        """
        super().__init__(verbosity)
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def log(self, timestamp, elapsedtime, step, data):
        self._logger.log(
            level=self._level,
            msg='{step} {data}'.format(
                step=step,
                data=', '.join(f'{k}: {v}' for k, v in data.items())
            )
        )

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        """ For simplicity this logger ignores metadata. """

    def flush(self):
        pass
