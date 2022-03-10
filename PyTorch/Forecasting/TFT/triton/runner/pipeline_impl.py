# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from .pipeline import Pipeline

pipeline = Pipeline()
pipeline.model_export(
    commands=(
        r"""
        if [[ "${EXPORT_FORMAT}" == "ts-trace" || "${EXPORT_FORMAT}" == "ts-script" ]]; then
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
            \
            --checkpoint ${CHECKPOINT_DIR}/ \
            --precision ${EXPORT_PRECISION} \
            \
            --dataloader triton/dataloader.py \
            --dataset ${DATASETS_DIR}/${DATASET} \
            --batch-size 1
        """,
    )
)
pipeline.model_conversion(
    commands=(
        r"""
        if [[ "${EXPORT_FORMAT}" == "ts-trace" || "${EXPORT_FORMAT}" == "ts-script" ]]; then
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
            --container-version 21.08 \
            --max-workspace-size 10000000000 \
            --atol target__0=100 \
            --rtol target__0=100
        """,
    )
)

pipeline.model_deploy(
    commands=(
        r"""
        if [[ "${FORMAT}" == "ts-trace" || "${FORMAT}" == "ts-script" ]]; then
            export CONFIG_FORMAT="torchscript"
        else
            export CONFIG_FORMAT="${FORMAT}"
        fi
        
        model-navigator triton-config-model \
            --model-repository ${MODEL_REPOSITORY_PATH} \
            --model-name ${MODEL_NAME} \
            --model-version 1 \
            --model-path ${SHARED_DIR}/converted_model \
            --model-format ${CONFIG_FORMAT} \
            --model-control-mode ${TRITON_LOAD_MODEL_METHOD} \
            --load-model \
            --load-model-timeout-s 100 \
            --verbose \
            \
            --backend-accelerator ${ACCELERATOR} \
            --tensorrt-precision ${PRECISION} \
            --tensorrt-capture-cuda-graph \
            --tensorrt-max-workspace-size 10000000000 \
            --max-batch-size ${MAX_BATCH_SIZE} \
            --batching dynamic \
            --preferred-batch-sizes ${TRITON_PREFERRED_BATCH_SIZES} \
            --max-queue-delay-us ${TRITON_MAX_QUEUE_DELAY} \
            --engine-count-per-device ${DEVICE}=${TRITON_GPU_ENGINE_COUNT}
        """,
    )
)
pipeline.triton_prepare_performance_profiling_data(
    commands=(
        r"""
        mkdir -p ${SHARED_DIR}/input_data
        """,
r"""
        python triton/prepare_input_data.py \
            --input-data-dir ${SHARED_DIR}/input_data/ \
            --dataset ${DATASETS_DIR}/${DATASET} \
            --checkpoint ${CHECKPOINT_DIR}/ \
        """,
    )
)
pipeline.triton_performance_offline_tests(
    commands=(
        r"""
        python triton/run_performance_on_triton.py \
            --model-repository ${MODEL_REPOSITORY_PATH} \
            --model-name ${MODEL_NAME} \
            --input-data ${SHARED_DIR}/input_data/data.json \
            --batch-sizes ${BATCH_SIZE} \
            --number-of-triton-instances ${TRITON_INSTANCES} \
            --batching-mode static \
            --evaluation-mode offline \
            --measurement-request-count ${REQUEST_COUNT} \
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
            --input-data ${SHARED_DIR}/input_data/data.json \
            --batch-sizes ${BATCH_SIZE} \
            --number-of-triton-instances ${TRITON_INSTANCES} \
            --number-of-model-instances ${TRITON_GPU_ENGINE_COUNT} \
            --batching-mode dynamic \
            --evaluation-mode online \
            --measurement-request-count 500 \
            --warmup \
            --performance-tool perf_analyzer \
            --result-path ${SHARED_DIR}/triton_performance_online.csv
        """,
    ),
    result_path="${SHARED_DIR}/triton_performance_online.csv",
)