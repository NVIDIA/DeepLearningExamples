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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

LOGGER = logging.getLogger(__name__)
DATALOADER_FN_NAME = "get_dataloader_fn"
GET_MODEL_FN_NAME = "get_model"
GET_SERVING_INPUT_RECEIVER_FN = "get_serving_input_receiver_fn"
GET_ARGPARSER_FN_NAME = "update_argparser"


class TensorSpec(NamedTuple):
    name: str
    dtype: str
    shape: Tuple


class Parameter(Enum):
    def __lt__(self, other: "Parameter") -> bool:
        return self.value < other.value


class Accelerator(Parameter):
    AMP = "amp"
    CUDA = "cuda"
    TRT = "trt"


class Precision(Parameter):
    FP16 = "fp16"
    FP32 = "fp32"
    TF32 = "tf32"  # Deprecated


class Format(Parameter):
    TF_GRAPHDEF = "tf-graphdef"
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    TF_ESTIMATOR = "tf-estimator"
    TF_KERAS = "tf-keras"
    ONNX = "onnx"
    TRT = "trt"
    TS_SCRIPT = "ts-script"
    TS_TRACE = "ts-trace"
    PYT = "pyt"


class Model(NamedTuple):
    handle: object
    precision: Optional[Precision]
    inputs: Dict[str, TensorSpec]
    outputs: Dict[str, TensorSpec]


def load_from_file(file_path, label, target):
    spec = importlib.util.spec_from_file_location(name=label, location=file_path)
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)  # pytype: disable=attribute-error
    return getattr(my_module, target, None)


class BaseLoader(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod
    def load(self, model_path: Union[str, Path], **kwargs) -> Model:
        """
        Loads and process model from file based on given set of args
        """
        pass


class BaseSaver(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod
    def save(self, model: Model, model_path: Union[str, Path]) -> None:
        """
        Save model to file
        """
        pass


class BaseRunner(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod
    def init_inference(self, model: Model):
        raise NotImplementedError


class BaseRunnerSession(abc.ABC):
    def __init__(self, model: Model):
        self._model = model

    @abc.abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, x: Dict[str, object]):
        raise NotImplementedError()

    def _set_env_variables(self) -> Dict[str, object]:
        """this method not remove values; fix it if needed"""
        to_set = {}
        old_values = {k: os.environ.pop(k, None) for k in to_set}
        os.environ.update(to_set)
        return old_values

    def _recover_env_variables(self, old_envs: Dict[str, object]):
        for name, value in old_envs.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = str(value)


class BaseConverter(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod
    def convert(self, model: Model, dataloader_fn) -> Model:
        raise NotImplementedError()

    @staticmethod
    def required_source_model_precision(requested_model_precision: Precision) -> Precision:
        return requested_model_precision


class BaseMetricsCalculator(abc.ABC):
    required_fn_name_for_signature_parsing: Optional[str] = None

    @abc.abstractmethod
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


class ShapeSpec(NamedTuple):
    min: Tuple
    opt: Tuple
    max: Tuple
