#!/usr/bin/env python3

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

import argparse
import csv
import logging
import os
import pathlib
import shutil
from distutils.version import LooseVersion
from enum import Enum
from importlib.metadata import version
from typing import Any, Dict, List

import yaml

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .deployment_toolkit.core import BatchingMode, EvaluationMode, MeasurementMode, OfflineMode, PerformanceTool
from .deployment_toolkit.model_analyzer import ModelAnalyzer, ModelAnalyzerConfig, ModelAnalyzerMode
from .deployment_toolkit.perf_analyzer import PerfAnalyzer, PerfAnalyzerConfig
from .deployment_toolkit.report import save_results, show_results, sort_results
from .deployment_toolkit.utils import parse_server_url
from .deployment_toolkit.warmup import performance_evaluation_warmup

LOGGER = logging.getLogger("run_performance_on_triton")

TRITON_CLIENT_VERSION = LooseVersion(version("tritonclient"))


def _log_dict(title: str, dict_: Dict[str, Any]):
    LOGGER.info(title)
    for key, value in dict_.items():
        LOGGER.info(f"\t{key} = {value}")


def _calculate_average_latency(r):
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
    avg_latency = sum([int(r.get(f, 0)) for f in avg_sum_fields])

    return avg_latency


def _update_performance_data(results: List, batch_size: int, performance_partial_file: str):
    row: Dict = {"Batch": batch_size}
    with open(performance_partial_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            avg_latency = _calculate_average_latency(r)
            row = {**row, **r, "avg latency": avg_latency}
            results.append(row)


def _model_analyzer_evaluation(
    server_url: str,
    model_name: str,
    input_data: str,
    input_shapes: List[str],
    batch_sizes: List[int],
    number_of_triton_instances: int,
    number_of_model_instances: int,
    measurement_mode: MeasurementMode,
    measurement_interval: int,
    measurement_request_count: int,
    concurrency_steps: int,
    batching_mode: BatchingMode,
    evaluation_mode: EvaluationMode,
    offline_mode: OfflineMode,
    model_repository: str,
    result_path: str,
    output_shared_memory_size: int = 102400,
    verbose: bool = False,
):
    _log_dict(
        "Selected configuration",
        {
            "server_url": server_url,
            "model_name": model_name,
            "input_data": input_data,
            "input_shapes": input_shapes,
            "batch_sizes": batch_sizes,
            "number_of_triton_instances": number_of_triton_instances,
            "number_of_model_instances": number_of_model_instances,
            "measurement_mode": measurement_mode,
            "measurement_interval": measurement_interval,
            "measurement_request_count": measurement_request_count,
            "concurrency_steps": concurrency_steps,
            "batching_mode": batching_mode,
            "evaluation_mode": evaluation_mode,
            "offline_mode": offline_mode,
            "output_shared_memory_size": output_shared_memory_size,
            "model_repository": model_repository,
            "result_path": result_path,
            "verbose": verbose,
        },
    )

    perf_analyzer_config = {
        "input-data": input_data,
        "measurement-interval": measurement_interval,
    }

    if TRITON_CLIENT_VERSION >= LooseVersion("2.11.0"):
        perf_analyzer_config["measurement-mode"] = measurement_mode.value
        perf_analyzer_config["measurement-request-count"] = measurement_request_count

    if evaluation_mode == EvaluationMode.OFFLINE:
        perf_analyzer_config["shared-memory"] = offline_mode.value
        perf_analyzer_config["output-shared-memory-size"] = output_shared_memory_size

    if input_shapes:
        perf_analyzer_config["shape"] = input_shapes[0]
        LOGGER.warning("Model Analyzer support only single shape param for Perf Analyzer.")

    if batching_mode == BatchingMode.STATIC:
        batch_sizes = batch_sizes
        concurrency = [number_of_triton_instances]
    elif batching_mode == BatchingMode.DYNAMIC:
        max_batch_size = max(batch_sizes)
        max_total_requests = 2 * max_batch_size * number_of_triton_instances * number_of_model_instances
        max_concurrency = min(256, max_total_requests)
        step = max(1, max_concurrency // concurrency_steps)
        min_concurrency = step

        concurrency = {"start": min_concurrency, "stop": max_concurrency, "step": step}
        batch_sizes = [max(1, max_total_requests // 256)]
    else:
        raise ValueError(f"Unsupported batching mode: {batching_mode}")

    protocol, host, port = parse_server_url(server_url)

    checkpoints = pathlib.Path("./checkpoints")
    if checkpoints.is_dir():
        shutil.rmtree(checkpoints.as_posix())

    checkpoints.mkdir(parents=True, exist_ok=True)

    config = {
        "model_repository": model_repository,
        "triton_launch_mode": "remote",
        "run_config_search_disable": True,
        "perf_analyzer_flags": perf_analyzer_config,
        "perf_analyzer_timeout": 3600,  # Workaround for Perf Analyzer timeout - use 1h
        "profile_models": [model_name],
        "batch_sizes": batch_sizes,
        "concurrency": concurrency,
        "verbose": verbose,
        "checkpoint_directory": checkpoints.as_posix(),
        "override_output_model_repository": True,
        "client_protocol": protocol,
        f"triton_{protocol}_endpoint": f"{host}:{port}",
    }

    if verbose:
        _log_dict("Model Analyzer profiling configuration", config)

    with open("config.yaml", "w") as file:
        yaml.safe_dump(config, file)

    config = ModelAnalyzerConfig()
    model_analyzer = ModelAnalyzer(config=config)
    model_analyzer.run(mode=ModelAnalyzerMode.PROFILE, verbose=verbose)

    result_path = pathlib.Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    for file in checkpoints.iterdir():
        if not file.is_file() or file.suffix != ".ckpt":
            continue

        LOGGER.info(f"Moving checkpoint {file.name} to {result_path}")
        shutil.move(file, result_path / file.name)

    inference_output_fields = [
        "batch_size",
        "concurrency",
        "perf_throughput",
        "perf_latency",
        "perf_client_send_recv",
        "perf_client_response_wait",
        "perf_server_queue",
        "perf_server_compute_input",
        "perf_server_compute_infer",
        "perf_server_compute_output",
    ]
    gpu_output_fields = [
        "gpu_uuid",
        "batch_size",
        "concurrency",
        "gpu_used_memory",
        "gpu_free_memory",
        "gpu_utilization",
        "gpu_power_usage",
    ]

    filename_model_inference = "metrics-model-inference.csv"
    filename_model_gpu = "metrics-model-gpu.csv"

    config = {
        "analysis_models": model_name,
        "checkpoint_directory": result_path.as_posix(),
        "export_path": "/tmp",
        "inference_output_fields": inference_output_fields,
        "gpu_output_fields": gpu_output_fields,
        "filename_model_inference": filename_model_inference,
        "filename_model_gpu": filename_model_gpu,
        "summarize": False,
    }

    if verbose:
        _log_dict("Model Analyzer analysis configuration", config)

    with open("config.yaml", "w") as file:
        yaml.safe_dump(config, file)

    config = ModelAnalyzerConfig()

    model_analyzer = ModelAnalyzer(config=config)
    model_analyzer.run(mode=ModelAnalyzerMode.ANALYZE, verbose=verbose)

    inference_metrics_file = pathlib.Path("/tmp") / "results" / filename_model_inference
    gpu_metrics_file = pathlib.Path("/tmp") / "results" / filename_model_gpu

    for file in [inference_metrics_file, gpu_metrics_file]:
        LOGGER.info(f"Moving metrics {file.name} to {result_path}")
        shutil.move(file, result_path / file.name)


def _perf_analyzer_evaluation(
    server_url: str,
    model_name: str,
    input_data: str,
    input_shapes: List[str],
    batch_sizes: List[int],
    number_of_triton_instances: int,
    number_of_model_instances: int,
    measurement_mode: MeasurementMode,
    measurement_interval: int,
    measurement_request_count: int,
    concurrency_steps: int,
    batching_mode: BatchingMode,
    evaluation_mode: EvaluationMode,
    offline_mode: OfflineMode,
    result_path: str,
    output_shared_memory_size: int = 102400,
    verbose: bool = False,
):
    protocol, host, port = parse_server_url(server_url)

    if batching_mode == BatchingMode.STATIC:
        batch_sizes = batch_sizes
        max_concurrency = 1
        min_concurrency = 1
        step = 1
    elif batching_mode == BatchingMode.DYNAMIC:
        max_batch_size = max(batch_sizes)
        max_total_requests = 2 * max_batch_size * number_of_triton_instances * number_of_model_instances
        max_concurrency = min(256, max_total_requests)
        step = max(1, max_concurrency // concurrency_steps)
        min_concurrency = step
        batch_sizes = [max(1, max_total_requests // 256)]
    else:
        raise ValueError(f"Unsupported batching mode: {batching_mode}")

    _log_dict(
        "Selected configuration",
        {
            "server_url": server_url,
            "model_name": model_name,
            "input_data": input_data,
            "input_shapes": input_shapes,
            "batch_sizes": batch_sizes,
            "number_of_triton_instances": number_of_triton_instances,
            "number_of_model_instances": number_of_model_instances,
            "measurement_mode": measurement_mode,
            "measurement_interval": measurement_interval,
            "measurement_request_count": measurement_request_count,
            "concurrency_steps": concurrency_steps,
            "batching_mode": batching_mode,
            "evaluation_mode": evaluation_mode,
            "offline_mode": offline_mode,
            "output_shared_memory_size": output_shared_memory_size,
            "result_path": result_path,
            "verbose": verbose,
        },
    )

    results: List[Dict] = list()
    for batch_size in batch_sizes:
        for concurrency in range(min_concurrency, max_concurrency + step, step):
            performance_partial_file = f"triton_performance_{evaluation_mode.value.lower()}_{batching_mode.value.lower()}_partial_{batch_size}_{concurrency}.csv"

            params = {
                "model-name": model_name,
                "model-version": 1,
                "batch-size": batch_size,
                "url": f"{host}:{port}",
                "protocol": protocol,
                "input-data": input_data,
                "measurement-interval": measurement_interval,
                "concurrency-range": f"{concurrency}:{concurrency}:1",
                "latency-report-file": performance_partial_file,
            }

            if verbose:
                params["extra-verbose"] = True

            if TRITON_CLIENT_VERSION >= LooseVersion("2.11.0"):
                params["measurement-mode"] = measurement_mode.value
                params["measurement-request-count"] = measurement_request_count

            if evaluation_mode == EvaluationMode.OFFLINE:
                params["shared-memory"] = offline_mode.value
                params["output-shared-memory-size"] = output_shared_memory_size

            if verbose:
                _log_dict(f"Perf Analyzer config for batch_size: {batch_size} and concurrency: {concurrency}", params)

            config = PerfAnalyzerConfig()
            for param, value in params.items():
                config[param] = value

            for shape in input_shapes:
                config["shape"] = shape

            perf_analyzer = PerfAnalyzer(config=config)
            perf_analyzer.run()
            _update_performance_data(results, batch_size, performance_partial_file)
            os.remove(performance_partial_file)

    results = sort_results(results=results)

    save_results(filename=result_path, data=results)
    show_results(results=results)


def _run_performance_analysis(
    server_url: str,
    model_name: str,
    input_data: str,
    input_shapes: List[str],
    batch_sizes: List[int],
    number_of_triton_instances: int,
    number_of_model_instances: int,
    measurement_mode: MeasurementMode,
    measurement_interval: int,
    measurement_request_count: int,
    concurrency_steps: int,
    batching_mode: BatchingMode,
    evaluation_mode: EvaluationMode,
    offline_mode: OfflineMode,
    output_shared_memory_size: int,
    performance_tool: PerformanceTool,
    model_repository: str,
    result_path: str,
    warmup: bool,
    verbose: bool,
):
    log_level = logging.INFO if not verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    if warmup:
        LOGGER.info("Running warmup before the main test")
        performance_evaluation_warmup(
            server_url=server_url,
            model_name=model_name,
            input_data=input_data,
            input_shapes=input_shapes,
            batch_sizes=batch_sizes,
            number_of_triton_instances=number_of_triton_instances,
            number_of_model_instances=number_of_model_instances,
            measurement_mode=measurement_mode,
            measurement_interval=measurement_interval,
            measurement_request_count=measurement_request_count,
            batching_mode=batching_mode,
            evaluation_mode=evaluation_mode,
            offline_mode=offline_mode,
            output_shared_memory_size=output_shared_memory_size,
        )

    if performance_tool == PerformanceTool.MODEL_ANALYZER:
        LOGGER.info("Using Model Analyzer for performance evaluation")
        _model_analyzer_evaluation(
            server_url=server_url,
            model_name=model_name,
            input_data=input_data,
            input_shapes=input_shapes,
            batch_sizes=batch_sizes,
            number_of_triton_instances=number_of_triton_instances,
            number_of_model_instances=number_of_model_instances,
            measurement_mode=measurement_mode,
            measurement_interval=measurement_interval,
            measurement_request_count=measurement_request_count,
            concurrency_steps=concurrency_steps,
            batching_mode=batching_mode,
            evaluation_mode=evaluation_mode,
            offline_mode=offline_mode,
            output_shared_memory_size=output_shared_memory_size,
            model_repository=model_repository,
            result_path=result_path,
            verbose=verbose,
        )
    elif performance_tool == PerformanceTool.PERF_ANALYZER:
        LOGGER.info("Using Perf Analyzer for performance evaluation")
        _perf_analyzer_evaluation(
            server_url=server_url,
            model_name=model_name,
            input_data=input_data,
            input_shapes=input_shapes,
            batch_sizes=batch_sizes,
            number_of_triton_instances=number_of_triton_instances,
            number_of_model_instances=number_of_model_instances,
            measurement_mode=measurement_mode,
            measurement_interval=measurement_interval,
            measurement_request_count=measurement_request_count,
            concurrency_steps=concurrency_steps,
            batching_mode=batching_mode,
            evaluation_mode=evaluation_mode,
            offline_mode=offline_mode,
            output_shared_memory_size=output_shared_memory_size,
            result_path=result_path,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported performance tool {performance_tool}")


class MeasurementMode(Enum):
    """
    Available measurement stabilization modes
    """

    COUNT_WINDOWS = "count_windows"
    TIME_WINDOWS = "time_windows"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        type=str,
        required=False,
        default="grpc://127.0.0.1:8001",
        help="Url to Triton server",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to test",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=False,
        default="random",
        help="Input data to perform profiling.",
    )
    parser.add_argument(
        "--input-shapes",
        action="append",
        required=False,
        help="Input data shape in form INPUT_NAME:<full_shape_without_batch_axis>.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        required=True,
        help="List of batch sizes to tests. Comma separated.",
    )
    parser.add_argument(
        "--number-of-triton-instances",
        type=int,
        default=1,
        help="Number of Triton Server instances",
    )
    parser.add_argument(
        "--number-of-model-instances",
        type=int,
        default=1,
        help="Number of models instances on Triton Server",
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
        required=False,
        help="Time window perf_analyzer will wait to stabilize the measurement",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--measurement-request-count",
        required=False,
        help="Number of samples on which perf_analyzer will stabilize the measurement",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--concurrency-steps",
        help="Define number of concurrency steps used for dynamic batching tests",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--batching-mode",
        choices=[item.value for item in BatchingMode],
        default=BatchingMode.STATIC.value,
        type=str,
        help="Select batching mode "
        "'static' run static batching scenario. "
        "'dynamic' run dynamic batching scenario.",
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
        default=100240,
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
    parser.add_argument("--result-path", type=str, required=True, help="Path where results files is stored.")
    parser.add_argument(
        "--warmup", help="Enable model warmup before performance test", action="store_true", default=False
    )
    parser.add_argument("-v", "--verbose", help="Verbose logs", action="store_true", default=False)

    args = parser.parse_args()

    batch_sizes = list(map(lambda x: int(x), args.batch_sizes.split(",")))
    _run_performance_analysis(
        server_url=args.server_url,
        model_name=args.model_name,
        input_data=args.input_data,
        input_shapes=args.input_shapes or [],
        batch_sizes=batch_sizes,
        number_of_triton_instances=args.number_of_triton_instances,
        number_of_model_instances=args.number_of_model_instances,
        measurement_mode=MeasurementMode(args.measurement_mode),
        measurement_interval=args.measurement_interval,
        measurement_request_count=args.measurement_request_count,
        concurrency_steps=args.concurrency_steps,
        batching_mode=BatchingMode(args.batching_mode),
        evaluation_mode=EvaluationMode(args.evaluation_mode),
        offline_mode=OfflineMode(args.offline_mode),
        output_shared_memory_size=args.output_shared_memory_size,
        performance_tool=PerformanceTool(args.performance_tool),
        model_repository=args.model_repository,
        result_path=args.result_path,
        warmup=args.warmup,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
