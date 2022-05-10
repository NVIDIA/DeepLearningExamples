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
import csv
import logging
import os
import pathlib
import sys
from distutils.version import LooseVersion
from typing import Dict, List, Optional

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ...core import EvaluationMode, MeasurementMode, OfflineMode
from ...report import save_results, show_results, sort_results
from ...utils import log_dict, parse_server_url
from .perf_analyzer import PerfAnalyzer
from .perf_config import PerfAnalyzerConfig

if LooseVersion(sys.version) >= LooseVersion("3.8.0"):
    from importlib.metadata import version

    TRITON_CLIENT_VERSION = LooseVersion(version("tritonclient"))
else:
    import pkg_resources

    TRITON_CLIENT_VERSION = LooseVersion(pkg_resources.get_distribution("tritonclient").version)

LOGGER = logging.getLogger("triton_performance_runner.perf_analyzer")


class PerfAnalyzerRunner:
    def __init__(
        self,
        server_url: str,
        model_name: str,
        input_data: str,
        input_shapes: List[str],
        batch_sizes: List[int],
        concurrency: List[int],
        measurement_mode: MeasurementMode,
        measurement_interval: int,
        measurement_request_count: int,
        evaluation_mode: EvaluationMode,
        offline_mode: OfflineMode,
        result_path: pathlib.Path,
        output_shared_memory_size: int = 102400,
        timeout: Optional[int] = None,
        verbose: bool = False,
    ):
        log_dict(
            "Selected configuration",
            {
                "server_url": server_url,
                "model_name": model_name,
                "input_data": input_data,
                "input_shapes": input_shapes,
                "batch_sizes": batch_sizes,
                "concurrency": concurrency,
                "measurement_mode": measurement_mode,
                "measurement_interval": measurement_interval,
                "measurement_request_count": measurement_request_count,
                "evaluation_mode": evaluation_mode,
                "offline_mode": offline_mode,
                "output_shared_memory_size": output_shared_memory_size,
                "result_path": result_path,
                "timeout": timeout,
                "verbose": verbose,
            },
        )

        if result_path.suffix != ".csv":
            raise ValueError(
                "Results path for Perf Analyzer is invalid. Please, provide the CSV file name. Example: results.csv"
            )

        self._server_url = server_url
        self._model_name = model_name
        self._input_data = input_data
        self._input_shapes = input_shapes
        self._batch_sizes = batch_sizes
        self._concurrency = concurrency
        self._measurement_mode = measurement_mode
        self._measurement_interval = measurement_interval
        self._measurement_request_count = measurement_request_count
        self._evaluation_mode = evaluation_mode
        self._offline_mode = offline_mode
        self._result_path = result_path
        self._output_shared_memory_size = output_shared_memory_size
        self._timeout = timeout
        self._verbose = verbose

        self._protocol, self._host, self._port = parse_server_url(server_url)

    def run(self):

        results: List[Dict] = []
        for batch_size in self._batch_sizes:
            for concurrency in self._concurrency:
                performance_partial_file = (
                    f"{self._evaluation_mode.value.lower()}_partial_{batch_size}_{concurrency}.csv"
                )

                params = {
                    "model-name": self._model_name,
                    "model-version": 1,
                    "batch-size": batch_size,
                    "url": f"{self._host}:{self._port}",
                    "protocol": self._protocol.value,
                    "input-data": self._input_data,
                    "measurement-interval": self._measurement_interval,
                    "concurrency-range": f"{concurrency}:{concurrency}:1",
                    "latency-report-file": performance_partial_file,
                }

                if self._verbose:
                    params["extra-verbose"] = True

                if TRITON_CLIENT_VERSION >= LooseVersion("2.11.0"):
                    params["measurement-mode"] = self._measurement_mode.value
                    params["measurement-request-count"] = self._measurement_request_count

                if self._evaluation_mode == EvaluationMode.OFFLINE:
                    params["shared-memory"] = self._offline_mode.value
                    params["output-shared-memory-size"] = self._output_shared_memory_size

                if self._verbose:
                    log_dict(
                        f"Perf Analyzer config for batch_size: {batch_size} and concurrency: {concurrency}", params
                    )

                config = PerfAnalyzerConfig()
                for param, value in params.items():
                    config[param] = value

                for shape in self._input_shapes:
                    config["shape"] = shape

                perf_analyzer = PerfAnalyzer(config=config, timeout=self._timeout)
                perf_analyzer.run()
                self._update_performance_data(results, batch_size, performance_partial_file)
                os.remove(performance_partial_file)

        results = sort_results(results=results)

        save_results(filename=self._result_path.as_posix(), data=results)
        show_results(results=results)

    def _calculate_average_latency(self, r):
        avg_sum_fields = [
            "Client Send",
            "Network+Server Send/Recv",
            "Server Queue",
            "Server Compute",
            "Server Compute Input",
            "Server Compute Infer",
            "Server Compute Output",
            "Client Recv",
        ]
        avg_latency = sum(int(r.get(f, 0)) for f in avg_sum_fields)

        return avg_latency

    def _update_performance_data(self, results: List, batch_size: int, performance_partial_file: str):
        row: Dict = {"Batch": batch_size}
        with open(performance_partial_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for r in reader:
                avg_latency = self._calculate_average_latency(r)
                row = {**row, **r, "avg latency": avg_latency}
                results.append(row)
