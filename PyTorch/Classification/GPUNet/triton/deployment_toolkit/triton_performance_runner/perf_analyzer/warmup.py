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
import logging
import pathlib
from distutils.version import LooseVersion
from importlib.metadata import version
from typing import List, Optional

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ...core import EvaluationMode, MeasurementMode, OfflineMode
from ...utils import parse_server_url
from .perf_analyzer import PerfAnalyzer
from .perf_config import PerfAnalyzerConfig

LOGGER = logging.getLogger("warmup")

TRITON_CLIENT_VERSION = LooseVersion(version("tritonclient"))


class PerfAnalyzerWarmupRunner:
    def __init__(
        self,
        server_url: str,
        model_name: str,
        batch_sizes: List[int],
        concurrency: List[int],
        input_data: str,
        input_shapes: List[str],
        measurement_mode: MeasurementMode,
        measurement_interval: int,
        measurement_request_count: int,
        offline_mode: OfflineMode,
        evaluation_mode: EvaluationMode,
        output_shared_memory_size: int,
        timeout: Optional[int],
    ):
        self._model_name = model_name
        self._input_data = input_data
        self._input_shapes = input_shapes
        self._measurement_mode = measurement_mode
        self._offline_mode = offline_mode
        self._evaluation_mode = evaluation_mode
        self._output_shared_memory_size = output_shared_memory_size

        self._protocol, self._host, self._port = parse_server_url(server_url)

        self._measurement_interval = 2 * measurement_interval
        self._measurement_request_count = 2 * measurement_request_count

        self._batch_sizes = [min(batch_sizes)]
        self._concurrency = [max(concurrency)]
        self._timeout = timeout

    def run(self):
        for batch_size in self._batch_sizes:
            for concurrency in self._concurrency:
                params = {
                    "model-name": self._model_name,
                    "model-version": 1,
                    "batch-size": batch_size,
                    "url": f"{self._host}:{self._port}",
                    "protocol": self._protocol.value,
                    "input-data": self._input_data,
                    "measurement-interval": self._measurement_interval,
                    "concurrency-range": f"{concurrency}:{concurrency}:1",
                    "verbose": True,
                }

                if TRITON_CLIENT_VERSION >= LooseVersion("2.11.0"):
                    params["measurement-mode"] = self._measurement_mode.value
                    params["measurement-request-count"] = self._measurement_request_count

                if self._evaluation_mode == EvaluationMode.OFFLINE:
                    params["shared-memory"] = self._offline_mode.value
                    params["output-shared-memory-size"] = self._output_shared_memory_size

                config = PerfAnalyzerConfig()
                for param, value in params.items():
                    config[param] = value

                for shape in self._input_shapes:
                    config["shape"] = shape

                perf_analyzer = PerfAnalyzer(config=config, timeout=self._timeout)
                perf_analyzer.run()
