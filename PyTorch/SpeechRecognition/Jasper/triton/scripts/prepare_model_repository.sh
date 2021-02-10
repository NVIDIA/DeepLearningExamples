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


# Create folder deploy/model_repo that will be used by TRITON

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/..
DEPLOY_DIR=${PROJECT_DIR}/deploy
HOST_REPO=${DEPLOY_DIR}/model_repo


MODELS_TENSORRT=${MODELS_TENSORRT:-"jasper-tensorrt jasper-tensorrt-ensemble"}
MODELS_TS_TRACE=${MODELS_TS_TRACE:-"jasper-ts-trace jasper-ts-trace-ensemble"}
MODELS_ONNX=${MODELS_ONNX:-"jasper-onnx jasper-onnx-ensemble"}
DECODERS="decoder-ts-script"
EXTRACTORS="feature-extractor-ts-trace"

MODELS=${MODELS:-"${MODELS_ONNX} ${MODELS_TENSORRT} ${MODELS_TS_TRACE}"}
PRECISION=${PRECISION:-"fp16" "fp32"}

# only link working models to install directory
rm -fr ${HOST_REPO} && mkdir -p ${HOST_REPO}
if [ -f /.dockerenv ]; then # inside docker
    chmod -R a+w ${HOST_REPO}
fi

echo "Setting up model repo at ${HOST_REPO}, models: ${MODELS} ..."
for m  in ${EXTRACTORS} ${DECODERS} ${MODELS}; do

    mkdir -p ${HOST_REPO}/$m
    cp ${PROJECT_DIR}/model_repo_configs/${PRECISION}/$m/config.pbtxt ${HOST_REPO}/$m/
    if [ -d "${PROJECT_DIR}/model_repo/${PRECISION}/$m/1"  ]; then
	echo "Creating symlink ls -sf /model_repo/${PRECISION}/$m/1 ${HOST_REPO}/$m"
	ln -sf /model_repo/${PRECISION}/$m/1 ${HOST_REPO}/$m
    else
	mkdir -p ${HOST_REPO}/$m/1
    fi
    if [ -f /.dockerenv ]; then # inside docker
	chmod -R a+w ${HOST_REPO}/$m
    fi
done
