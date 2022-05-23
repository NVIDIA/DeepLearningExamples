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
        python3 triton/export_model.py \
            --input-path triton/model.py \
            --input-type tf-keras \
            --output-path ${SHARED_DIR}/exported_model.savedmodel \
            --output-type ${EXPORT_FORMAT} \
            --ignore-unknown-parameters \
            \
            --checkpoint-dir ${CHECKPOINT_DIR}/checkpoint \
            --batch-size ${MAX_BATCH_SIZE} \
            --precision ${EXPORT_PRECISION} \
            \
            --dataloader triton/dataloader.py \
            --batch-size ${MAX_BATCH_SIZE} \
            --data-pattern "${DATASETS_DIR}/outbrain/valid/*.parquet"
        """,
    )
)
pipeline.model_conversion(
    commands=(
        r"""
        model-navigator convert \
            --model-name ${MODEL_NAME} \
            --model-path ${SHARED_DIR}/exported_model.savedmodel \
            --output-path ${SHARED_DIR}/converted_model \
            --target-formats ${FORMAT} \
            --target-precisions ${PRECISION} \
            --launch-mode local \
            --override-workspace \
            --verbose \
            \
            --onnx-opsets 13 \
            --max-batch-size ${MAX_BATCH_SIZE} \
            --max-workspace-size 8589934592 \
            --atol wide_deep_model=0.015 \
            --rtol wide_deep_model=12.0
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
            --load-model-timeout-s 120 \
            --verbose \
            \
            --batching ${MODEL_BATCHING} \
            --backend-accelerator ${BACKEND_ACCELERATOR} \
            --tensorrt-precision ${PRECISION} \
            --tensorrt-capture-cuda-graph \
            --max-batch-size ${MAX_BATCH_SIZE} \
            --preferred-batch-sizes ${MAX_BATCH_SIZE} \
            --engine-count-per-device ${DEVICE_KIND}=${NUMBER_OF_MODEL_INSTANCES}
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
            --batch-sizes ${MEASUREMENT_OFFLINE_BATCH_SIZES} \
            --concurrency ${MEASUREMENT_OFFLINE_CONCURRENCY} \
            --performance-tool ${PERFORMANCE_TOOL} \
            --measurement-request-count 100 \
            --evaluation-mode offline \            
            --warmup \
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
            --batch-sizes ${MEASUREMENT_ONLINE_BATCH_SIZES} \
            --concurrency ${MEASUREMENT_ONLINE_CONCURRENCY} \
            --performance-tool ${PERFORMANCE_TOOL} \
            --measurement-request-count 500 \
            --evaluation-mode online \
            --warmup \
            --result-path ${SHARED_DIR}/triton_performance_online.csv
        """,
    ),
    result_path="${SHARED_DIR}/triton_performance_online.csv",
)