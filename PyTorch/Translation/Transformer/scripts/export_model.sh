#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

NV_VISIBLE_DEVICES=${1:-"0"}
DOCKER_BRIDGE=${2:-"host"}
checkpoint=${3:-"/checkpoints/checkpoint_jit.pt"}
batch_size=${4:-"5120"}
WORKSPACE=${5:-"/workspace/translation"}
triton_model_version=${6:-1}
triton_model_name=${7:-"transformer"}
triton_dyn_batching_delay=${8:-0}
triton_engine_count=${9:-1}
triton_model_overwrite=${10:-"False"}

DEPLOYER="deployer.py"

#TODO: add fp16 option

CMD="python triton/${DEPLOYER} \
    --ts-script \
    --save-dir ${WORKSPACE}/triton/triton_models \
    --triton-model-name ${triton_model_name} \
    --triton-model-version ${triton_model_version} \
    --triton-max-batch-size ${batch_size} \
    --triton-dyn-batching-delay ${triton_dyn_batching_delay} \
    --triton-engine-count ${triton_engine_count} "

ENCODER_EXPORT_CMD="$CMD --triton-model-name ${triton_model_name}-encoder"
DECODER_EXPORT_CMD="$CMD --triton-model-name ${triton_model_name}-decoder"

MODEL_ARGS=" -- --checkpoint ${checkpoint} \
    --batch-size=${batch_size} \
    --num-batches=2 \
    --data /data "

ENCODER_EXPORT_CMD+="${MODEL_ARGS} --part encoder"
DECODER_EXPORT_CMD+="${MODEL_ARGS} --part decoder"

echo Exporting encoder...
bash scripts/docker/launch.sh "${ENCODER_EXPORT_CMD}" ${NV_VISIBLE_DEVICES} ${DOCKER_BRIDGE}
echo Exporting decoder...
bash scripts/docker/launch.sh "${DECODER_EXPORT_CMD}" ${NV_VISIBLE_DEVICES} ${DOCKER_BRIDGE}
