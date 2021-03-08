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

MODEL_TYPE=${1:-"ts-trace"}
DATA_DIR=${2} # folder with data
FILE=${3} # json manifest file, OR single wav file

JASPER_CONTAINER_TAG=${JASPER_CONTAINER_TAG:-jasper:triton}

if [ "$#" -ge 1 ] && [ "${FILE: -4}" == ".wav" ]; then 
  CMD="python /jasper/triton/jasper-client.py --data_dir /data --audio_filename ${FILE} --model_platform ${MODEL_TYPE}"
  ARGS=""
  ARGS="$ARGS -v $DATA_DIR:/data"
elif [ "$#" -ge 1 ] && [ "${FILE: -4}" == "json" ]; then
  ARGS=""
  ARGS="$ARGS -v $DATA_DIR:/data"
  CMD="python /jasper/triton/jasper-client.py --manifest_filename ${FILE} --model_platform ${MODEL_TYPE} --data_dir /data"
else
  ARGS="-it"
  CMD=""
fi

echo "========== STARTING ${JASPER_CONTAINER_TAG} =========="

set -x
nvidia-docker run --rm -it \
   --net=host \
   --shm-size=1g \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -v ${PROJECT_DIR}:/jasper \
   --name=jasper-triton-client \
   ${ARGS} ${JASPER_CONTAINER_TAG} ${CMD}
set +x
