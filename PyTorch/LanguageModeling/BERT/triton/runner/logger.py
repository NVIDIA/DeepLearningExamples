# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import pathlib

import coloredlogs


class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level=level)
        self._file_path = None

    def initialize(self, file_path: pathlib.Path):
        self._file_path = file_path

    def write(self, log: str):
        if not self._file_path:
            return

        with open(self._file_path, "+a") as file:
            file.write(log)


LOGGER = Logger("runner")

log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(format=log_format)
coloredlogs.install(
    level=logging.INFO,
    fmt=log_format,
    logger=LOGGER,
    field_styles={
        "asctime": {"color": "green"},
        "hostname": {"color": "magenta"},
        "levelname": {"bold": True, "color": "blue"},
        "name": {"color": "blue"},
        "programname": {"color": "cyan"},
        "username": {"color": "yellow"},
    },
    reconfigure=True,
)
