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
import shutil
import sys
from distutils.version import LooseVersion
from typing import List, Optional

import yaml

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ...core import EvaluationMode, MeasurementMode, OfflineMode
from ...utils import log_dict, parse_server_url
from .model_analyzer import ModelAnalyzer, ModelAnalyzerMode
from .model_analyzer_config import ModelAnalyzerConfig

if LooseVersion(sys.version) >= LooseVersion("3.8.0"):
    from importlib.metadata import version

    TRITON_CLIENT_VERSION = LooseVersion(version("tritonclient"))
    TRITON_MODEL_ANALYZER_VERSION = LooseVersion(version("triton-model-analyzer"))
else:
    import pkg_resources

    TRITON_CLIENT_VERSION = LooseVersion(pkg_resources.get_distribution("tritonclient").version)
    TRITON_MODEL_ANALYZER_VERSION = LooseVersion(pkg_resources.get_distribution("triton-model-analyzer").version)

LOGGER = logging.getLogger("triton_performance_runner.model_analyzer")


class ModelAnalyzerRunner:
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
        model_repository: str,
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
                "model_repository": model_repository,
                "result_path": result_path,
                "verbose": verbose,
            },
        )

        if result_path.suffix:
            raise ValueError(
                "Results path for Model Analyzer is invalid. Please, provide the directory name. Example: results"
            )

        self._checkpoints = pathlib.Path("./checkpoints")
        self._result_path = result_path
        self._verbose = verbose

        self._filename_model_inference = "metrics-model-inference.csv"
        self._filename_model_gpu = "metrics-model-gpu.csv"

        self._profile_config = self._prepare_profile_config(
            server_url=server_url,
            model_name=model_name,
            input_data=input_data,
            input_shapes=input_shapes,
            batch_sizes=batch_sizes,
            concurrency=concurrency,
            measurement_mode=measurement_mode,
            measurement_interval=measurement_interval,
            measurement_request_count=measurement_request_count,
            evaluation_mode=evaluation_mode,
            offline_mode=offline_mode,
            model_repository=model_repository,
            output_shared_memory_size=output_shared_memory_size,
            checkpoints=self._checkpoints,
            verbose=verbose,
        )
        self._analyze_config = self._prepare_analyze_config(
            model_name=model_name,
            result_path=result_path,
            verbose=verbose,
            filename_model_inference=self._filename_model_inference,
            filename_model_gpu=self._filename_model_gpu,
        )

    def run(self):
        self._result_path.mkdir(parents=True, exist_ok=True)

        if self._checkpoints.is_dir():
            shutil.rmtree(self._checkpoints.as_posix())
        self._checkpoints.mkdir(parents=True, exist_ok=True)

        model_analyzer = ModelAnalyzer(config=self._profile_config)
        model_analyzer.run(mode=ModelAnalyzerMode.PROFILE, verbose=self._verbose)

        for file in self._checkpoints.iterdir():
            if not file.is_file() or file.suffix != ".ckpt":
                continue

            LOGGER.info(f"Moving checkpoint {file.name} to {self._result_path}")
            shutil.move(file, self._result_path / file.name)

        model_analyzer = ModelAnalyzer(config=self._analyze_config)
        model_analyzer.run(mode=ModelAnalyzerMode.ANALYZE, verbose=self._verbose)

        inference_metrics_file = pathlib.Path("/tmp") / "results" / self._filename_model_inference
        gpu_metrics_file = pathlib.Path("/tmp") / "results" / self._filename_model_gpu

        for file in [inference_metrics_file, gpu_metrics_file]:
            LOGGER.info(f"Moving metrics {file.name} to {self._result_path}")
            shutil.move(file, self._result_path / file.name)

    def _prepare_profile_config(
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
        model_repository: str,
        checkpoints: pathlib.Path,
        output_shared_memory_size: int = 102400,
        verbose: bool = False,
    ):
        protocol, host, port = parse_server_url(server_url)

        perf_analyzer_config = self._perf_analyzer_config(
            input_data,
            input_shapes,
            measurement_mode,
            measurement_interval,
            measurement_request_count,
            evaluation_mode,
            offline_mode,
            output_shared_memory_size,
        )

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
            "client_protocol": protocol.value,
            f"triton_{protocol.value}_endpoint": f"{host}:{port}",
        }

        if verbose:
            log_dict("Model Analyzer profiling configuration", config)

        with open("config_profile.yaml", "w") as file:
            yaml.safe_dump(config, file)

        config = ModelAnalyzerConfig()
        config["config-file"] = "config_profile.yaml"

        return config

    def _prepare_analyze_config(
        self,
        model_name: str,
        result_path: pathlib.Path,
        filename_model_inference: str,
        filename_model_gpu: str,
        verbose: bool,
    ):
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
            log_dict("Model Analyzer analysis configuration", config)

        with open("config_analyze.yaml", "w") as file:
            yaml.safe_dump(config, file)

        config = ModelAnalyzerConfig()
        config["config-file"] = "config_analyze.yaml"

        return config

    def _perf_analyzer_config(
        self,
        input_data: str,
        input_shapes: List[str],
        measurement_mode: MeasurementMode,
        measurement_interval: int,
        measurement_request_count: int,
        evaluation_mode: EvaluationMode,
        offline_mode: OfflineMode,
        output_shared_memory_size: int = 102400,
    ):
        perf_analyzer_config = {
            "measurement-interval": measurement_interval,
        }

        if TRITON_MODEL_ANALYZER_VERSION >= LooseVersion("1.8.0"):
            perf_analyzer_config["input-data"] = [input_data]
        else:
            perf_analyzer_config["input-data"] = input_data

        if TRITON_CLIENT_VERSION >= LooseVersion("2.11.0"):
            perf_analyzer_config["measurement-mode"] = measurement_mode.value
            perf_analyzer_config["measurement-request-count"] = measurement_request_count

        if evaluation_mode == EvaluationMode.OFFLINE:
            perf_analyzer_config["shared-memory"] = offline_mode.value
            perf_analyzer_config["output-shared-memory-size"] = output_shared_memory_size

        if input_shapes:
            if TRITON_MODEL_ANALYZER_VERSION > LooseVersion("1.8.0"):
                perf_analyzer_config["shape"] = input_shapes
            else:
                perf_analyzer_config["shape"] = input_shapes[0]
                LOGGER.warning("Model Analyzer <= 1.8.0 support only single shape param for Perf Analyzer.")

        return perf_analyzer_config
