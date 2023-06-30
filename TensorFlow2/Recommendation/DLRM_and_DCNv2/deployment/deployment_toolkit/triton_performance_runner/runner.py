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
# method from PEP-366 to support relative import in executed modules
import logging
import pathlib
from typing import List, Optional, Dict, Tuple

if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ..core import EvaluationMode, MeasurementMode, OfflineMode
from .perf_analyzer import PerfAnalyzerRunner, PerfAnalyzerWarmupRunner

LOGGER = logging.getLogger("triton_performance_runner")


class TritonPerformanceRunner:
    def __init__(
        self,
        server_url: str,
        model_name: str,
        input_data: Dict[int, Tuple],
        batch_sizes: List[int],
        concurrency: List[int],
        measurement_mode: MeasurementMode,
        measurement_interval: int,
        measurement_request_count: int,
        evaluation_mode: EvaluationMode,
        offline_mode: OfflineMode,
        output_shared_memory_size: int,
        result_path: pathlib.Path,
        warmup: bool,
        timeout: Optional[int],
        verbose: bool,
        flattened_input: bool,
    ):

        self._warmup_runner = None
        if warmup:
            LOGGER.info("Running warmup before the main test")
            self._warmup_runner = PerfAnalyzerWarmupRunner(
                server_url=server_url,
                model_name=model_name,
                input_data=input_data,
                batch_sizes=batch_sizes,
                concurrency=concurrency,
                measurement_mode=measurement_mode,
                measurement_interval=measurement_interval,
                measurement_request_count=measurement_request_count,
                evaluation_mode=evaluation_mode,
                offline_mode=offline_mode,
                output_shared_memory_size=output_shared_memory_size,
                timeout=timeout,
                flattened_input=flattened_input
            )

        LOGGER.info("Using Perf Analyzer for performance evaluation")
        self._runner = PerfAnalyzerRunner(
            server_url=server_url,
            model_name=model_name,
            input_data=input_data,
            batch_sizes=batch_sizes,
            measurement_mode=measurement_mode,
            measurement_interval=measurement_interval,
            measurement_request_count=measurement_request_count,
            concurrency=concurrency,
            evaluation_mode=evaluation_mode,
            offline_mode=offline_mode,
            output_shared_memory_size=output_shared_memory_size,
            result_path=result_path,
            timeout=timeout,
            verbose=verbose,
            flattened_input=flattened_input
        )

    def run(self):
        if self._warmup_runner:
            self._warmup_runner.run()

        self._runner.run()
