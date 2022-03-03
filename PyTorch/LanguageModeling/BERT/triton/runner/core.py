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
import pathlib
from enum import Enum
from typing import Any, Dict, List

import yaml


class CustomDumper(yaml.Dumper):
    """
    Custom YAML dumper to avoid craeting aliases
    """

    def ignore_aliases(self, data: Dict) -> bool:
        return True


class Paths:
    """
    Paths mapping inside Triton Container
    """

    MODEL_REPOSITORY_PATH = "/mnt/triton-models"
    LIBRARIES_PATH = "/mnt/libs"


class Framework(Enum):
    """
    Supported frameworks
    """

    TensorFlow1 = "TensorFlow1"
    TensorFlow2 = "TensorFlow2"
    PyTorch = "PyTorch"


class Command:
    """Represents wrapper of raw string command"""

    def __init__(self, data: str):
        """
        Store command data
        Args:
            data: string with bash commands to execute
        """
        self._data = data

    def __str__(self) -> str:
        """
        String object representation

        Returns:
            String
        """
        return self._data


class DataObject(object):
    """
    Data object representation handling recursive transformation from object to dict
    """

    READ_ONLY = set()

    def to_dict(self) -> Dict:
        """
        Represent object as dictionary

        Returns:
            Dict
        """
        data = dict()
        filtered_data = {key: value for key, value in self.__dict__.items() if key not in self.READ_ONLY}
        for key, value in filtered_data.items():
            data[key] = self._convert_value(value)

        return data

    def _convert_value(self, value: Any) -> Any:
        """
        Convert value based on its type

        Args:
            value: variable to convert

        Returns:
            Converted object
        """
        if isinstance(value, DataObject):
            value = value.to_dict()
        elif isinstance(value, dict):
            value = self._from_dict(value)
        elif isinstance(value, list):
            value = self._from_list(value)
        elif isinstance(value, Enum):
            value = value.value
        elif isinstance(value, pathlib.Path):
            value = value.as_posix()

        return value

    def _from_dict(self, values: Dict) -> Any:
        """
        Convert dictionary values

        Args:
            values: dictionary with values

        Returns:
            Any
        """
        data = dict()
        for key, value in values.items():
            data[key] = self._convert_value(value)

        return data

    def _from_list(self, values: List) -> Any:
        """
        Convert list of values

        Args:
            values: list with values

        Returns:
            Any
        """
        items = list()
        for value in values:
            item = self._convert_value(value)
            items.append(item)

        return items


AVAILABLE_FRAMEWORKS = [f.value for f in Framework]
