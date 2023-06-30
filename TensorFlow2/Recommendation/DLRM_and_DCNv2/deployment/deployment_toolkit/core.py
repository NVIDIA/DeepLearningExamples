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
import importlib
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

LOGGER = logging.getLogger(__name__)
DATALOADER_FN_NAME = "get_dataloader_fn"
GET_MODEL_FN_NAME = "get_model"
GET_SERVING_INPUT_RECEIVER_FN = "get_serving_input_receiver_fn"
GET_ARGPARSER_FN_NAME = "update_argparser"


def load_from_file(file_path, label, target):
    spec = importlib.util.spec_from_file_location(name=label, location=file_path)
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)  # pytype: disable=attribute-error
    return getattr(my_module, target, None)


class BaseMetricsCalculator(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    def calc(
        self,
        *,
        ids: List[Any],
        y_pred: Dict[str, np.ndarray],
        x: Optional[Dict[str, np.ndarray]],
        y_real: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, float]:
        """
        Calculates error/accuracy metrics
        Args:
            ids: List of ids identifying each sample in the batch
            y_pred: model output as dict where key is output name and value is output value
            x: model input as dict where key is input name and value is input value
            y_real: input ground truth as dict where key is output name and value is output value
        Returns:
            dictionary where key is metric name and value is its value
        """
        pass

    @abc.abstractmethod
    def update(
        self,
        ids: List[Any],
        y_pred: Dict[str, np.ndarray],
        x: Optional[Dict[str, np.ndarray]],
        y_real: Optional[Dict[str, np.ndarray]],
    ):
        pass

    @property
    @abc.abstractmethod
    def metrics(self) -> Dict[str, Any]:
        pass


class ShapeSpec(NamedTuple):
    min: Tuple
    opt: Tuple
    max: Tuple


class MeasurementMode(Enum):
    """
    Available measurement stabilization modes
    """

    COUNT_WINDOWS = "count_windows"
    TIME_WINDOWS = "time_windows"


class EvaluationMode(Enum):
    """
    Available evaluation modes
    """

    OFFLINE = "offline"
    ONLINE = "online"


class OfflineMode(Enum):
    """
    Available offline mode for memory
    """

    SYSTEM = "system"
    CUDA = "cuda"
