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


# method from PEP-366 to support relative import in executed modules
import argparse
import logging
from pathlib import Path
from typing import List

if __name__ == "__main__" and __package__ is None:
    __package__ = Path(__file__).parent.name

from .benchmark.benchmark import Benchmark
from .benchmark.checkpoints import HttpCheckpoint
from .benchmark.core import LOGGER
from .benchmark.executor import DockerExecutor
from .deployment_toolkit.core import Accelerator, Format, Precision

AVAILABLE_MODEL_FORMATS = [f.value for f in Format]
AVAILABLE_MODEL_PRECISIONS = [p.value for p in Precision]
AVAILABLE_MODEL_ACCELERATORS = [a.value for a in Accelerator]

def run_benchmark(
        devices: List[str],
        model_name: str,
        model_version: int,
        model_format: str,
        container_version: str,
        checkpoint: str,
        max_batch_size: int,
        precision: str,
        number_of_model_instances: int,
        preferred_batch_sizes: List[int],
        max_queue_delay_us: int,
        backend_accelerator: str,
        verbose: bool,
        **kwargs
):
    benchmark = Benchmark(
        devices=devices,
        model_name=model_name,
        model_version=model_version,
        framework="TensorFlow1",
        container_version=container_version,
        checkpoint=HttpCheckpoint(checkpoint),
        verbose=verbose
    )
    benchmark.model_conversion(
        cmds=(
            r"""
        python3 triton/convert_model.py \
            --input-path triton/rn50_model.py \
            --input-type tf-estimator \
            --output-path ${SHARED_DIR}/model \
            --output-type ${FORMAT} \
            --onnx-opset 12 \
            --onnx-optimized 1 \
            --max-batch-size ${MAX_BATCH_SIZE} \
            --max-workspace-size 4294967296 \
            --ignore-unknown-parameters \
            \
            --model-dir ${CHECKPOINT_DIR} \
            --precision ${PRECISION} \
            --dataloader triton/dataloader.py \
            --data-dir ${DATASETS_DIR}/imagenet
        """,
        )
    )

    benchmark.model_deploy(
        cmds=(
            r"""
        python3 triton/deploy_model.py \
            --model-repository ${MODEL_REPOSITORY_PATH} \
            --model-path ${SHARED_DIR}/model \
            --model-format ${FORMAT} \
            --model-name ${MODEL_NAME} \
            --model-version 1 \
            --max-batch-size ${MAX_BATCH_SIZE} \
            --precision ${PRECISION} \
            --number-of-model-instances ${NUMBER_OF_MODEL_INSTANCES} \
            --max-queue-delay-us ${TRITON_MAX_QUEUE_DELAY} \
            --preferred-batch-sizes ${TRITON_PREFERRED_BATCH_SIZES} \
            --capture-cuda-graph 0 \
            --backend-accelerator ${BACKEND_ACCELERATOR} \
            --load-model ${TRITON_LOAD_MODEL_METHOD}
        """,
        )
    )
    benchmark.triton_performance_offline_tests(
        cmds=(
            r"""
        python triton/run_offline_performance_test_on_triton.py \
            --server-url ${TRITON_SERVER_URL} \
            --model-name ${MODEL_NAME} \
            --number-of-warmup-iterations 5 \
            --input-data random \
            --batch-sizes ${BATCH_SIZE} \
            --triton-instances ${TRITON_INSTANCES} \
            --result-path ${SHARED_DIR}/triton_performance_offline.csv
        """,
        ),
        result_path="${SHARED_DIR}/triton_performance_offline.csv",
    )
    benchmark.triton_performance_online_tests(
        cmds=(
            r"""
        python triton/run_online_performance_test_on_triton.py \
            --server-url ${TRITON_SERVER_URL} \
            --model-name ${MODEL_NAME} \
            --number-of-warmup-iterations 5 \
            --input-data random \
            --batch-sizes ${BATCH_SIZE} \
            --triton-instances ${TRITON_INSTANCES} \
            --number-of-model-instances ${NUMBER_OF_MODEL_INSTANCES} \
            --result-path ${SHARED_DIR}/triton_performance_online.csv
        """,
        ),
        result_path="${SHARED_DIR}/triton_performance_online.csv",
    )

    benchmark.configuration(
        precision=precision,
        max_batch_size=max_batch_size,
        format=model_format,
        accelerator=backend_accelerator,
        triton_gpu_engine_count=number_of_model_instances,
        triton_preferred_batch_sizes=preferred_batch_sizes,
        triton_max_queue_delay_us=max_queue_delay_us,
        **kwargs
    )

    executor = DockerExecutor()
    executor.run(benchmark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark for model.")
    parser.add_argument("--devices", help="NVIDIA GPU device ID on which Triton Inference Server is ran. Accept multiple values", nargs="*", required=False)
    parser.add_argument("--model-name", help="Model name. Default: ResNet50", default="ResNet50", required=False)
    parser.add_argument("--model-version", default="1", help="Version of model. Default: 1", required=False)
    parser.add_argument("--checkpoint", default="https://api.ngc.nvidia.com/v2/models/nvidia/rn50_tf_amp_ckpt/versions/20.06.0/zip", help="Checkpoint url. Default: https://api.ngc.nvidia.com/v2/models/nvidia/rn50_tf_amp_ckpt/versions/20.06.0/zip", required=False)
    parser.add_argument("--container-version", help="Version of container for Triton Inference Server. Default: 20.12", default="20.12", required=False)
    parser.add_argument(
        "--model-format",
        choices=AVAILABLE_MODEL_FORMATS,
        help="Format of exported model. Default: tf-savedmodel",
        default="tf-savedmodel",
        required=False
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=AVAILABLE_MODEL_PRECISIONS,
        help="Model precision (parameter used only by Tensorflow backend with TensorRT optimization). Default: fp16",
        required=False
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Batch size used for benchmark. Maximal batch size which is used to convert model. Default: 32",
        required=False
    )
    parser.add_argument(
        "--number-of-model-instances",
        type=int,
        default=2,
        help="Number of model instances per GPU (model instances). Default: 2",
        required=False
    )
    parser.add_argument(
        "--preferred-batch-sizes",
        type=int,
        nargs="*",
        help="Batch sizes that the dynamic batching should attempt to create. "
             "In case --max-queue-delay-us is set and this parameter is not, default value will be calculated based on --max-batch-size",
        required=False
    )
    parser.add_argument(
        "--max-queue-delay-us",
        type=int,
        default=100,
        help="Max delay time which dynamic batch shall wait to form a batch. Default: 100",
        required=False
    )
    parser.add_argument(
        "--backend-accelerator",
        choices=AVAILABLE_MODEL_ACCELERATORS,
        type=str,
        default="cuda",
        help="Select backend accelerator used for model. Default: cuda",
        required=False
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Provide verbose output")

    args = parser.parse_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    LOGGER.setLevel(log_level)

    LOGGER.info(f"args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    run_benchmark(**vars(args))
