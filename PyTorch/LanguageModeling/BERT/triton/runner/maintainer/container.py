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
import abc
from typing import Any


class Container(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self._container = None

    @abc.abstractmethod
    def start(self):
        """
        Start container
        """
        pass

    @abc.abstractmethod
    def stop(self):
        """
        Stop container
        """

    @abc.abstractmethod
    def run(self, command: str) -> Any:
        """
        Run command inside container
        Args:
            command: command to execute

        Returns:
            Any
        """
        pass
