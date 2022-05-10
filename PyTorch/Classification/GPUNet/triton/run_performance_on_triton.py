#!/usr/bin/env python3

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

import argparse
import logging
import pathlib

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .deployment_toolkit.core import EvaluationMode, MeasurementMode, OfflineMode, PerformanceTool
from .deployment_toolkit.triton_performance_runner import TritonPerformanceRunner

LOGGER = logging.getLogger("run_performance_on_triton")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to test",
    )
    parser.add_argument(
        "--result-path",
        type=pathlib.Path,
        required=True,
        help="Path where results files is stored.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Url to Triton server",
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default=1,
        help="Version of model",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="random",
        help="Input data to perform profiling.",
    )
    parser.add_argument(
        "--input-shapes",
        action="append",
        help="Input data shape in form INPUT_NAME:<full_shape_without_batch_axis>.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        default=[1],
        help="List of batch sizes to tests.",
        nargs="*",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=[1],
        help="List of concurrency modes.",
        nargs="*",
    )
    parser.add_argument(
        "--measurement-mode",
        choices=[item.value for item in MeasurementMode],
        default=MeasurementMode.COUNT_WINDOWS.value,
        type=str,
        help="Select measurement mode "
        "'time_windows' stabilize performance on measurement window. "
        "'count_windows' stabilize performance on number of samples.",
    )
    parser.add_argument(
        "--measurement-interval",
        help="Time window perf_analyzer will wait to stabilize the measurement",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--measurement-request-count",
        help="Number of samples on which perf_analyzer will stabilize the measurement",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=[item.value for item in EvaluationMode],
        default=EvaluationMode.OFFLINE.value,
        type=str,
        help="Select evaluation mode "
        "'offline' run offline analysis and use GPU memory to pass tensors. "
        "'online' run online analysis and use HTTP protocol.",
    )
    parser.add_argument(
        "--offline-mode",
        choices=[item.value for item in OfflineMode],
        default=OfflineMode.SYSTEM.value,
        type=str,
        help="Select offline mode "
        "'system' pass tensors through CPU RAM memory. "
        "'cuda' pass tensors through GPU RAM memory.",
    )
    parser.add_argument(
        "--output-shared-memory-size",
        default=102400,
        type=int,
        help="Size of memory buffer allocated for output with dynamic shapes in bytes. "
        "Has to be equal to maximal size of output tensor.",
    )
    parser.add_argument(
        "--performance-tool",
        choices=[item.value for item in PerformanceTool],
        default=PerformanceTool.MODEL_ANALYZER.value,
        type=str,
        help="Select performance tool for measurement mode "
        "'model_analyzer' use Model Analyzer "
        "'perf_analyzer' use Perf Analyzer",
    )
    parser.add_argument(
        "--model-repository",
        default=None,
        type=str,
        help="Path to model repository. Valid when using Model Analyzer",
    )
    parser.add_argument(
        "--warmup",
        help="Enable model warmup before performance test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--timeout",
        help="Timeout for performance analysis",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose logs",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    runner = TritonPerformanceRunner(
        server_url=args.server_url,
        model_name=args.model_name,
        input_data=args.input_data,
        input_shapes=args.input_shapes or [],
        batch_sizes=args.batch_sizes,
        measurement_mode=MeasurementMode(args.measurement_mode),
        measurement_interval=args.measurement_interval,
        measurement_request_count=args.measurement_request_count,
        concurrency=args.concurrency,
        evaluation_mode=EvaluationMode(args.evaluation_mode),
        offline_mode=OfflineMode(args.offline_mode),
        output_shared_memory_size=args.output_shared_memory_size,
        performance_tool=PerformanceTool(args.performance_tool),
        model_repository=args.model_repository,
        result_path=args.result_path,
        warmup=args.warmup,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    runner.run()


if __name__ == "__main__":
    main()
