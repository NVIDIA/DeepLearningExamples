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
TRTIS_DIR=${SCRIPT_DIR}/..

DEPLOY_DIR=${DEPLOY_DIR:-${TRTIS_DIR}/deploy}
MODELS_DIR=${MODEL_DIR:-"$DEPLOY_DIR/model_repo"}
TRTIS_CONTAINER_TAG=${TRTIS_CONTAINER_TAG:-"nvcr.io/nvidia/tensorrtserver:19.09-py3"}
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}
TRTIS=${TRTIS:-jasper-trtis}

# Ensure that the server is closed when the script exits
function cleanup_server {
    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    logfile="/tmp/${TRTIS}-${current_time}.log"
    echo "Shutting down ${TRTIS} container, log is in ${logfile}"
    docker logs ${TRTIS} > ${logfile} 2>&1
    docker stop ${TRTIS} > /dev/null 2>&1
}

trap "exit" INT

if [ "$(docker inspect -f "{{.State.Running}}" ${TRTIS})" = "true" ]; then
    if [ "$1" == "norestart" ]; then
       echo "${TRTIS} is already running ..."
       exit 0
    fi   
    cleanup_server || true
fi

# To start  TRTIS container with alternative commandline, set CMD
CMD=${CMD:-"/opt/tensorrtserver/bin/trtserver --log-verbose=100 --exit-on-error=false --strict-model-config=false --model-store=/models"}
DAEMON=${DAEMON:-"-d"}
RM=${RM:-"--rm"}

set -x
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 \
       --runtime nvidia \
       -e NVIDIA_VISIBLE_DEVICES=${NV_VISIBLE_DEVICES} \
       -v ${MODELS_DIR}:/models \
       -v ${TRTIS_DIR}/model_repo:/model_repo \
       --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  \
       ${DAEMON} ${RM} --name ${TRTIS} ${TRTIS_CONTAINER_TAG} \
       ${CMD}
set +x
