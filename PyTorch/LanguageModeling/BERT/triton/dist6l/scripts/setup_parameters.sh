#!/usr/bin/env bash
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Setting up deployment parameters"

export FORMAT="onnx"
export PRECISION="fp16"
export EXPORT_FORMAT="onnx"
export EXPORT_PRECISION="fp16"
export ACCELERATOR="trt"
export ACCELERATOR_PRECISION="fp16"
export CAPTURE_CUDA_GRAPH="0"
export BATCH_SIZE="16"
export MAX_BATCH_SIZE="16"
export MAX_SEQ_LENGTH="384"
export CHECKPOINT_VARIANT="dist-6l-qa"
export CHECKPOINT_DIR=${CHECKPOINTS_DIR}/${CHECKPOINT_VARIANT}
export TRITON_MAX_QUEUE_DELAY="1"
export TRITON_GPU_ENGINE_COUNT="1"
export TRITON_PREFERRED_BATCH_SIZES="1"

if [[ "${FORMAT}" == "ts-trace" || "${FORMAT}" == "ts-script" ]]; then
    export CONFIG_FORMAT="torchscript"
else
    export CONFIG_FORMAT="${FORMAT}"
fi

if [[ "${EXPORT_FORMAT}" == "trt" ]]; then
    export FLAG="--fixed-batch-dim"
else
    export FLAG=""
fi

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

if [[ "${EXPORT_FORMAT}" == "ts-trace" || "${EXPORT_FORMAT}" == "ts-script" ]]; then
    export FORMAT_SUFFIX="pt"
else
    export FORMAT_SUFFIX="${EXPORT_FORMAT}"
fi