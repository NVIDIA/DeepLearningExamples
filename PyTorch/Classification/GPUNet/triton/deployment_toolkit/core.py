# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


class TensorSpec(NamedTuple):
    name: str
    dtype: str
    shape: Tuple


class Parameter(Enum):
    def __lt__(self, other: "Parameter") -> bool:
        return self.value < other.value

    def __str__(self):
        return self.value


class BackendAccelerator(Parameter):
    NONE = "none"
    AMP = "amp"
    TRT = "trt"


class ExportPrecision(Parameter):
    FP16 = "fp16"
    FP32 = "fp32"


class Precision(Parameter):
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


class DeviceKind(Parameter):
    CPU = "cpu"
    GPU = "gpu"


class ModelInputType(Parameter):
    TF_GRAPHDEF = "tf-graphdef"
    TF_ESTIMATOR = "tf-estimator"
    TF_KERAS = "tf-keras"
    PYT = "pyt"


class Format(Parameter):
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TRT = "trt"
    FASTERTRANSFORMER = "fastertransformer"

    # deprecated, backward compatibility only
    TS_TRACE = "ts-trace"
    TS_SCRIPT = "ts-script"


class ExportFormat(Parameter):
    TF_SAVEDMODEL = "tf-savedmodel"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"

    # deprecated, backward compatibility only
    TS_TRACE = "ts-trace"
    TS_SCRIPT = "ts-script"


class TorchJit(Parameter):
    NONE = "none"
    TRACE = "trace"
    SCRIPT = "script"


class Model(NamedTuple):
    handle: object
    # TODO: precision should be removed
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
    def save(self, model: Model, model_path: Union[str, Path], dataloader_fn) -> None:
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
        self._evaluations = []
        self._measurement = False

    @abc.abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, x: Dict[str, object]):
        raise NotImplementedError()

    def start_measurement(self):
        self._measurement = True
        self._evaluations = []

    def stop_measurement(self, batch_size: int = 1):
        LOGGER.info("Removing worst and best results")
        evaluations = sorted(self._evaluations)[2:-2]
        LOGGER.debug(f"Filtered: {evaluations}")
        average_latency_ms = sum(evaluations) / len(evaluations)
        LOGGER.debug(f"Average latency: {average_latency_ms:.2f} [ms]")
        throughput = (1000.0 / average_latency_ms) * batch_size
        LOGGER.debug(f"Throughput: {throughput:.2f} [infer/sec]")

        self._measurement = False

        return throughput, average_latency_ms

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


class TimeMeasurement:
    def __init__(self, session: BaseRunnerSession):
        self._session = session
        self._start = 0
        self._end = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._session._measurement:
            return

        self._end = time.time()
        diff = (self._end - self._start) * 1000.0
        LOGGER.debug(f"Iteration time {diff:.2f} [ms]")
        self._session._evaluations.append(diff)


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


class PerformanceTool(Enum):
    """
    Available performance evaluation tools
    """

    MODEL_ANALYZER = "model_analyzer"
    PERF_ANALYZER = "perf_analyzer"


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
