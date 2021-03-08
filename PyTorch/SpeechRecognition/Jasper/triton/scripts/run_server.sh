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
TRITON_DIR=${SCRIPT_DIR}/..

DEPLOY_DIR=${DEPLOY_DIR:-${TRITON_DIR}/deploy}
MODELS_DIR=${MODEL_DIR:-"$DEPLOY_DIR/model_repo"}
TRITON_CONTAINER_TAG=${TRITON_CONTAINER_TAG:-"nvcr.io/nvidia/tritonserver:20.10-py3"}
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"0"} # FIXME "all"
TRITON=${TRITON:-jasper-triton-server}

MODEL_TYPE=${1:-""}
PRECISION=${2:-""}

# Ensure that the server is closed when the script exits
function cleanup_server {
    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    logfile="/tmp/${TRITON}-${current_time}.log"
    echo "Shutting down ${TRITON} container, log is in ${logfile}"
    docker logs ${TRITON} > ${logfile} 2>&1
    docker stop ${TRITON} > /dev/null 2>&1
}

trap "exit" INT

# FIXME
# if [ "$(docker inspect -f "{{.State.Running}}" ${TRITON})" = "true" ]; then
#     if [ "$1" == "norestart" ]; then
#        echo "${TRITON} is already running ..."
#        exit 0
#     fi
#     cleanup_server || true
# fi

if [ -n "$MODEL_TYPE" ] && [ -n "$PRECISION" ]; then
    MODELS="jasper-${MODEL_TYPE} jasper-${MODEL_TYPE}-ensemble" PRECISION=${PRECISION} ${SCRIPT_DIR}/prepare_model_repository.sh
fi

# To start  TRITON container with alternative commandline, set CMD
CMD=${CMD:-"/opt/tritonserver/bin/tritonserver --log-verbose=100 --exit-on-error=true --strict-model-config=false --model-store=/models"}
DAEMON=${DAEMON:-"-d"}
RM=${RM:-"--rm"}

if [ ! -d "${MODELS_DIR}" ]; then
    echo "${MODELS_DIR} does not exist!"
    echo "Exiting from script ${0}."
    exit 1
fi

set -x
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 \
       --runtime nvidia \
       -e NVIDIA_VISIBLE_DEVICES=${NV_VISIBLE_DEVICES} \
       -v ${MODELS_DIR}:/models \
       -v ${TRITON_DIR}/model_repo:/model_repo \
       --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  \
       ${DAEMON} ${RM} --name ${TRITON} ${TRITON_CONTAINER_TAG} \
       ${CMD}
set +x
