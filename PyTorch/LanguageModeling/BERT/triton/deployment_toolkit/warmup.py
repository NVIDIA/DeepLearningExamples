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
from distutils.version import LooseVersion
from importlib.metadata import version
from typing import List

TRITON_CLIENT_VERSION = LooseVersion(version("tritonclient"))

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .core import BatchingMode, EvaluationMode, MeasurementMode, OfflineMode
from .perf_analyzer import PerfAnalyzer, PerfAnalyzerConfig
from .utils import parse_server_url

LOGGER = logging.getLogger("warmup")


def performance_evaluation_warmup(
    server_url: str,
    model_name: str,
    batch_sizes: List[int],
    number_of_triton_instances: int,
    number_of_model_instances: int,
    input_data: str,
    input_shapes: List[str],
    measurement_mode: MeasurementMode,
    measurement_interval: int,
    measurement_request_count: int,
    batching_mode: BatchingMode,
    offline_mode: OfflineMode,
    evaluation_mode: EvaluationMode,
    output_shared_memory_size: int,
):
    protocol, host, port = parse_server_url(server_url)

    measurement_interval = 2 * measurement_interval
    measurement_request_count = 2 * measurement_request_count

    if batching_mode == BatchingMode.STATIC:
        batch_sizes = sorted({1, batch_sizes[-1]})
        max_concurrency = 1
        min_concurrency = 1
        step = 1
    elif batching_mode == BatchingMode.DYNAMIC:
        max_batch_size = max(batch_sizes)
        max_total_requests = 2 * max_batch_size * number_of_triton_instances * number_of_model_instances
        max_concurrency = min(256, max_total_requests)
        step = max(1, max_concurrency // 2)
        min_concurrency = step
        batch_sizes = [max(1, max_total_requests // 256)]
    else:
        raise ValueError(f"Unsupported batching mode: {batching_mode}")

    for batch_size in batch_sizes:
        for concurrency in range(min_concurrency, max_concurrency + step, step):
            params = {
                "model-name": model_name,
                "model-version": 1,
                "batch-size": batch_size,
                "url": f"{host}:{port}",
                "protocol": protocol,
                "input-data": input_data,
                "measurement-interval": measurement_interval,
                "concurrency-range": f"{concurrency}:{concurrency}:1",
                "output-shared-memory-size": output_shared_memory_size,
            }

            if TRITON_CLIENT_VERSION >= LooseVersion("2.11.0"):
                params["measurement-mode"] = measurement_mode.value
                params["measurement-request-count"] = measurement_request_count

            if evaluation_mode == EvaluationMode.OFFLINE:
                params["shared-memory"] = offline_mode.value
                params["output-shared-memory-size"] = output_shared_memory_size

            config = PerfAnalyzerConfig()
            for param, value in params.items():
                config[param] = value

            for shape in input_shapes:
                config["shape"] = shape

            perf_analyzer = PerfAnalyzer(config=config)
            perf_analyzer.run()
