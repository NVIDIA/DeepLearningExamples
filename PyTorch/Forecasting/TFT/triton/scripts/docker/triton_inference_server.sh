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

# Install Docker
. /etc/os-release && \
curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
echo "deb [arch=amd64] https://download.docker.com/linux/debian buster stable" > /etc/apt/sources.list.d/docker.list && \
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey| apt-key add - && \
curl -s -L https://nvidia.github.io/nvidia-docker/$ID$VERSION_ID/nvidia-docker.list > /etc/apt/sources.list.d/nvidia-docker.list && \
apt-get update && \
apt-get install -y docker-ce docker-ce-cli containerd.io nvidia-docker2
WORKDIR="${WORKDIR:=$(pwd)}"
export DATASETS_DIR=${WORKDIR}/datasets
export WORKSPACE_DIR=${WORKDIR}/runner_workspace
export CHECKPOINTS_DIR=${WORKSPACE_DIR}/checkpoints
export MODEL_REPOSITORY_PATH=${WORKSPACE_DIR}/model_store
export SHARED_DIR=${WORKSPACE_DIR}/shared_dir
NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:=all}

docker run --rm -d \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
  -e ORT_TENSORRT_FP16_ENABLE=1 \
  -v ${MODEL_REPOSITORY_PATH}:${MODEL_REPOSITORY_PATH} \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  nvcr.io/nvidia/tritonserver:22.11-py3 tritonserver \
  --model-store=${MODEL_REPOSITORY_PATH} \
  --strict-model-config=false \
  --exit-on-error=true \
  --model-control-mode=explicit