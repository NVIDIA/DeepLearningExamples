#!/usr/bin/env bash
# Copyright (c) 2021-2022 NVIDIA CORPORATION. All rights reserved.
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
WORKDIR="${WORKDIR:=$(pwd)}"
export DATASETS_DIR=${WORKDIR}/datasets
export WORKSPACE_DIR=${WORKDIR}/runner_workspace
export CHECKPOINTS_DIR=${WORKSPACE_DIR}/checkpoints
export MODEL_REPOSITORY_PATH=${WORKSPACE_DIR}/model_store
export SHARED_DIR=${WORKSPACE_DIR}/shared_dir

echo "Preparing directories"
mkdir -p ${WORKSPACE_DIR}
mkdir -p ${DATASETS_DIR}
mkdir -p ${CHECKPOINTS_DIR}
mkdir -p ${MODEL_REPOSITORY_PATH}
mkdir -p ${SHARED_DIR}

echo "Setting up environment"
export MODEL_NAME=TFT
export ENSEMBLE_MODEL_NAME=
export TRITON_LOAD_MODEL_METHOD=explicit
export TRITON_INSTANCES=1