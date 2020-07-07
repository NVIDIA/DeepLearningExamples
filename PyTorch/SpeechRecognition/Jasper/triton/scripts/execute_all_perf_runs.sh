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

# This will run all the necessary scripts to generate ALL needed output


#### input arguments
ARCH=${ARCH:-75}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
RESULT_DIR=${RESULT_DIR}
CHECKPOINT=${CHECKPOINT:-"jasper_fp16.pt"}
MAX_SEQUENCE_LENGTH_FOR_ENGINE=${MAX_SEQUENCE_LENGTH_FOR_ENGINE:-1792}
####

for dir in $CHECKPOINT_DIR $RESULT_DIR; do
    if [[ $dir != /* ]]; then
        echo "All directory paths must be absolute paths!"
        echo "${dir} is not an absolute path"
        exit 1
    fi

    if [ ! -d $dir ]; then
        echo "All directory paths must exist!"
        echo "${dir} does not exist"
        exit 1
    fi
done

REGENERATE_ENGINES=${REGENERATE_ENGINES:-"yes"}
PRECISION_TESTS=${PRECISION_TESTS:-"fp16 fp32"}
export GPU=${GPU:-}

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/../..
MODEL_REPO=${MODEL_REPO:-"${PROJECT_DIR}/trtis/model_repo"}

# We need to make sure TRTIS uses only one GPU, same as export does
# for TRTIS
export NVIDIA_VISIBLE_DEVICES=${GPU}

export TRTIS_CLIENT_CONTAINER_TAG=tensorrtserver_client

trap "exit" INT


SCRIPT=${SCRIPT_DIR}/generate_perf_results.sh

function run_for_length () {
    TRTIS_CLIENT_CONTAINER_TAG=tensorrtserver_client AUDIO_LENGTH=$1 BATCH_SIZE=$2 RESULT_DIR=${RESULT_DIR} PRECISION=${PRECISION} ${SCRIPT} 
}


for PRECISION in ${PRECISION_TESTS};
do

    if [ "${REGENERATE_ENGINES}" == "yes" ]; then
        echo "REGENERATE_ENGINES==yes, forcing re-export"
    else	
        if [ -f ${MODEL_REPO}/jasper-onnx/1/jasper.onnx ]; then
            echo "Found ${MODEL_REPO}/jasper-onnx/1/jasper.onnx, skipping model export. Set REGENERATE_ENGINES=yes to force re-export"
        else
            REGENERATE_ENGINES=yes
            echo "${MODEL_REPO}/jasper-onnx/1/jasper.onnx not found, forcing re-export"
        fi
    fi
  
    if [ "${REGENERATE_ENGINES}" == "yes" ]; then
        ARCH=${ARCH} CHECKPOINT_DIR=${CHECKPOINT_DIR} CHECKPOINT=${CHECKPOINT} PRECISION=${PRECISION} MAX_SEQUENCE_LENGTH_FOR_ENGINE=${MAX_SEQUENCE_LENGTH_FOR_ENGINE} \
        ${PROJECT_DIR}/trtis/scripts/export_model.sh || exit 1
    fi
  
    for BATCH_SIZE in 1 2 4 8 16 32 64;
    do
        
        # 7 Seconds
        run_for_length 112000 $BATCH_SIZE

        # 2 seconds
        run_for_length 32000 $BATCH_SIZE

        # 16.7 Seconds
        run_for_length 267200 $BATCH_SIZE

    done
    # prepare for FP32 run
    REGENERATE_ENGINES=yes
done



