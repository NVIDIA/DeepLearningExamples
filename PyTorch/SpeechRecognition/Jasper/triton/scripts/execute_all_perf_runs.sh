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

if [ -f /.dockerenv ]; then # inside docker
    echo "The script \"$0\" script should be run from outside the container. Exiting."
    exit 1
fi

#### input arguments
RESULT_DIR=${RESULT_DIR}
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

export GPU=${GPU:-}

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/../..
MODEL_REPO=${MODEL_REPO:-"${PROJECT_DIR}/triton/deploy/model_repo"}

# We need to make sure TRITON uses only one GPU, same as export does
# for TRITON
export NVIDIA_VISIBLE_DEVICES=${GPU}
export TRITON_CLIENT_CONTAINER_TAG=jasper:triton

trap "exit" INT


function run_for_length () {
    TRITON_CLIENT_CONTAINER_TAG=jasper:triton \
			       AUDIO_LENGTH=$1 \
			       BATCH_SIZE=$2 \
			       RESULT_DIR=${RESULT_DIR} \
			       PRECISION=${PRECISION} \
			       ${SCRIPT_DIR}/generate_perf_results.sh
}

PRECISION_TESTS=${PRECISION_TESTS:-"fp16 fp32"}
BATCH_SIZES=${BATCH_SIZES:-"1" "2" "4" "8"}
SEQ_LENS=${SEQ_LENS:-"32000" "112000" "267200"} # i.e. ,7, 2, and 16.7 seconds

for PRECISION in ${PRECISION_TESTS};
do
    for BATCH_SIZE in ${BATCH_SIZES};
    do
	for LENGTH in ${SEQ_LENS};
	do
            run_for_length $LENGTH $BATCH_SIZE
	done
    done
done
