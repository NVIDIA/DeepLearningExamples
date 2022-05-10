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
import pathlib

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ...runner.pipeline import Pipeline

pipeline = Pipeline()
pipeline.model_export(
    commands=(
        r"""
        if [[ "${EXPORT_FORMAT}" == "torchscript" ]]; then
            export FORMAT_SUFFIX="pt"
        else
            export FORMAT_SUFFIX="${EXPORT_FORMAT}"
        fi
        python3 triton/export_model.py \
            --input-path triton/model.py \
            --input-type pyt \
            --output-path ${SHARED_DIR}/exported_model.${FORMAT_SUFFIX} \
            --output-type ${EXPORT_FORMAT} \
            --ignore-unknown-parameters \
            --onnx-opset 13 \
            --torch-jit ${TORCH_JIT} \
            \
            --config /workspace/gpunet/configs/batch1/GV100/0.65ms.json \
            --checkpoint ${CHECKPOINT_DIR}/0.65ms.pth.tar \
            --precision ${EXPORT_PRECISION} \
            \
            --dataloader triton/dataloader.py \
            --val-path ${DATASETS_DIR}/ \
            --is-prunet False \
            --batch-size 1
        """,
    )
)
pipeline.model_conversion(
    commands=(
        r"""
        if [[ "${EXPORT_FORMAT}" == "torchscript" ]]; then
            export FORMAT_SUFFIX="pt"
        else
            export FORMAT_SUFFIX="${EXPORT_FORMAT}"
        fi
        model-navigator convert \
            --model-name ${MODEL_NAME} \
            --model-path ${SHARED_DIR}/exported_model.${FORMAT_SUFFIX} \
            --output-path ${SHARED_DIR}/converted_model \
            --target-formats ${FORMAT} \
            --target-precisions ${PRECISION} \
            --launch-mode local \
            --override-workspace \
            --verbose \
            \
            --onnx-opsets 13 \
            --max-batch-size ${MAX_BATCH_SIZE} \
            --container-version 21.12 \
            --max-workspace-size 10000000000 \
            --atol OUTPUT__0=100 \
            --rtol OUTPUT__0=100
        """,
    )
)

pipeline.model_deploy(
    commands=(
        r"""
        model-navigator triton-config-model \
            --model-repository ${MODEL_REPOSITORY_PATH} \
            --model-name ${MODEL_NAME} \
            --model-version 1 \
            --model-path ${SHARED_DIR}/converted_model \
            --model-format ${FORMAT} \
            --model-control-mode explicit \
            --load-model \
            --load-model-timeout-s 100 \
            --verbose \
            \
            --backend-accelerator ${BACKEND_ACCELERATOR} \
            --tensorrt-precision ${PRECISION} \
            --tensorrt-capture-cuda-graph \
            --tensorrt-max-workspace-size 10000000000 \
            --max-batch-size ${MAX_BATCH_SIZE} \
            --batching ${MODEL_BATCHING} \
            --preferred-batch-sizes ${MAX_BATCH_SIZE} \
            --engine-count-per-device gpu=${NUMBER_OF_MODEL_INSTANCES}
        """,
    )
)
pipeline.triton_performance_offline_tests(
    commands=(
        r"""
        python triton/run_performance_on_triton.py \
            --model-repository ${MODEL_REPOSITORY_PATH} \
            --model-name ${MODEL_NAME} \
            --input-data random \
            --batch-sizes 1 2 4 8 16 32 64 \
            --concurrency 1 \
            --evaluation-mode offline \
            --measurement-request-count 10 \
            --warmup \
            --performance-tool perf_analyzer \
            --result-path ${SHARED_DIR}/triton_performance_offline.csv
        """,
    ),
    result_path="${SHARED_DIR}/triton_performance_offline.csv",
)
pipeline.triton_performance_online_tests(
    commands=(
        r"""
        python triton/run_performance_on_triton.py \
            --model-repository ${MODEL_REPOSITORY_PATH} \
            --model-name ${MODEL_NAME} \
            --input-data random \
            --batch-sizes 1 \
            --concurrency 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 \
            --evaluation-mode online \
            --measurement-request-count 500 \
            --warmup \
            --performance-tool perf_analyzer \
            --result-path ${SHARED_DIR}/triton_performance_online.csv
        """,
    ),
    result_path="${SHARED_DIR}/triton_performance_online.csv",
)