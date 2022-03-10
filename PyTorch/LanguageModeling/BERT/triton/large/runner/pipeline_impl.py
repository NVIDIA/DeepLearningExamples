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
import pathlib

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ...runner.pipeline import Pipeline

pipeline = Pipeline()
pipeline.model_export(
    commands=(
        r"""
        if [[ "${EXPORT_FORMAT}" == "ts-trace" || "${EXPORT_FORMAT}" == "ts-script" ]]; then
            export FORMAT_SUFFIX="pt"
        else
            export FORMAT_SUFFIX="${EXPORT_FORMAT}"
        fi
        if [[ "${EXPORT_FORMAT}" == "trt" ]]; then
            export FLAG="--fixed-batch-dim"
        else
            export FLAG=""
        fi
        python3 triton/export_model.py \
            --input-path triton/model.py \
            --input-type pyt \
            --output-path ${SHARED_DIR}/exported_model.${FORMAT_SUFFIX} \
            --output-type ${EXPORT_FORMAT} \
            --dataloader triton/dataloader.py \
            --ignore-unknown-parameters \
            --onnx-opset 13 \
            ${FLAG} \
            \
            --config-file bert_configs/large.json \
            --checkpoint ${CHECKPOINT_DIR}/bert_large_qa.pt \
            --precision ${EXPORT_PRECISION} \
            \
            --vocab-file ${DATASETS_DIR}/data/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt \
            --max-seq-length ${MAX_SEQ_LENGTH} \
            --predict-file ${DATASETS_DIR}/data/squad/v1.1/dev-v1.1.json \
            --batch-size ${MAX_BATCH_SIZE}
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
        if [ "${EXPORT_FORMAT}" != "${FORMAT}" ]; then
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
                --inputs input__0:${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH}:int32 \
                --inputs input__1:${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH}:int32 \
                --inputs input__2:${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH}:int32 \
                --min-shapes input__0=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                             input__1=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                             input__2=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                --max-shapes input__0=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                             input__1=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                             input__2=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                --opt-shapes input__0=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                             input__1=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                             input__2=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                --max-batch-size ${MAX_BATCH_SIZE} \
                --tensorrt-max-workspace-size 8589934592 \
                --atol 2 output__0=5.0 \
                         output__1=5.0 \
                --rtol 1 output__0=5.0 \
                         output__1=5.0 \
                | grep -v "broadcasting input1 to make tensors conform"
        else
            mv ${SHARED_DIR}/exported_model.${FORMAT_SUFFIX} ${SHARED_DIR}/converted_model
            mv ${SHARED_DIR}/exported_model.${FORMAT_SUFFIX}.yaml ${SHARED_DIR}/converted_model.yaml 2>/dev/null || true
        fi
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
        
        if [[ "${FORMAT}" == "trt" ]]; then
            export MBS="0"
        else
            export MBS="${MAX_BATCH_SIZE}"
        fi
        
        model-navigator triton-config-model \
            --model-repository ${MODEL_REPOSITORY_PATH} \
            --model-name ${MODEL_NAME} \
            --model-version 1 \
            --model-path ${SHARED_DIR}/converted_model \
            --model-format ${CONFIG_FORMAT} \
            --model-control-mode ${TRITON_LOAD_MODEL_METHOD} \
            --verbose \
            --load-model \
            --load-model-timeout-s 100 \
            \
            --backend-accelerator ${ACCELERATOR} \
            --tensorrt-precision ${ACCELERATOR_PRECISION}  \
            --max-batch-size ${MBS} \
            --preferred-batch-sizes ${TRITON_PREFERRED_BATCH_SIZES} \
            --max-queue-delay-us ${TRITON_MAX_QUEUE_DELAY} \
            --engine-count-per-device gpu=${TRITON_GPU_ENGINE_COUNT}
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
            --dataloader triton/dataloader.py \
            --input-data-dir ${SHARED_DIR}/input_data \
            \
            --batch-size ${MAX_BATCH_SIZE} \
            --max-seq-length ${MAX_SEQ_LENGTH} \
            --predict-file ${DATASETS_DIR}/data/squad/v1.1/dev-v1.1.json \
            --vocab-file ${DATASETS_DIR}/data/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt
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
            --input-shapes input__0:${MAX_SEQ_LENGTH} \
            --input-shapes input__1:${MAX_SEQ_LENGTH} \
            --input-shapes input__2:${MAX_SEQ_LENGTH} \
            --batch-sizes ${BATCH_SIZE} \
            --number-of-triton-instances ${TRITON_INSTANCES} \
            --number-of-model-instances ${TRITON_GPU_ENGINE_COUNT} \
            --batching-mode static \
            --evaluation-mode offline \
            --performance-tool perf_analyzer \
            --result-path ${SHARED_DIR}/triton_performance_offline.csv
        """,
    ),
    result_path="${SHARED_DIR}/triton_performance_offline.csv",
)
