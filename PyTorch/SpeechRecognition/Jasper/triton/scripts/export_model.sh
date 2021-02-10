#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/../..
TRITON_DIR=${PROJECT_DIR}/triton
export PYTHONPATH=$TRITON_DIR:$PYTHONPATH

# Jasper model arguments ---------------------------------------------
BATCH_SIZE=${BATCH_SIZE:-8}
DATA_DIR=${DATA_DIR:-"/datasets/LibriSpeech"}
DATASET=${DATASET:-"test-clean"}
MODEL_CONFIG=${MODEL_CONFIG:-"configs/jasper10x5dr_speedp-offline_speca_nomask.yaml"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/checkpoints/nvidia_jasper_210205.pt"}
EMA=${EMA:-true}

ARGS=" --batch_size $BATCH_SIZE "
ARGS+=" --dataset_dir $DATA_DIR "
ARGS+=" --val_manifest $DATA_DIR/librispeech-${DATASET}-wav.json "
ARGS+=" --model_config $MODEL_CONFIG  "
[ -n "$CHECKPOINT_PATH" ] && \
    ARGS+=" --ckpt=${CHECKPOINT_PATH}"
[ "$EMA" = true ] &&                 ARGS+=" --ema"
## -------------------------------------------------------------------

run_converter() {
    set -x
    mkdir -p $3
    chmod a+w $3
    python triton/converter.py --model-module jasper_module \
    	   --convert $1 \
    	   --precision ${PRECISION} \
    	   --save-dir $3 \
    	   -- \
    	   ${ARGS} --component $2
    set +x
}

CONVERTS=${CONVERTS:-"feature-extractor" "decoder" "ts-trace" "onnx" "tensorrt"}
CONVERT_PRECISIONS=${CONVERT_PRECISIONS:-"fp16" "fp32"}

for PRECISION in ${CONVERT_PRECISIONS[@]}; do

    MODEL_REPO_DIR="${TRITON_DIR}/model_repo/${PRECISION}/"
    mkdir -p $MODEL_REPO_DIR
    chmod a+w "${TRITON_DIR}" "${TRITON_DIR}/model_repo/" "${TRITON_DIR}/model_repo/${PRECISION}/"

    if [[ " ${CONVERTS[@]} " =~ " feature-extractor " ]]; then
    	## Export data preprocessor
    	run_converter "ts-trace" "feature-extractor" \
		      ${MODEL_REPO_DIR}/feature-extractor-ts-trace/1
    fi
    if [[ " ${CONVERTS[@]} " =~ " decoder " ]]; then
    	## Export greedy decoder
    	run_converter "ts-script" "decoder" \
		      ${MODEL_REPO_DIR}/decoder-ts-script/1
    fi
    BACKENDS=( "ts-trace" "onnx" "tensorrt" )
    for BACKEND in ${BACKENDS[@]}; do
	if [[ " ${CONVERTS[@]} " =~ " ${BACKEND} " ]]; then
            ## Export acoustic model - Jasper
            run_converter "${BACKEND}" "model" \
			  ${MODEL_REPO_DIR}/jasper-${BACKEND}/1
	fi
    done
done
