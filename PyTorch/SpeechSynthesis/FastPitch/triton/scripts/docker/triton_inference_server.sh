#!/usr/bin/env bash
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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

NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:=all}

docker run --rm -d \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
  -v ${MODEL_REPOSITORY_PATH}:${MODEL_REPOSITORY_PATH} \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/tritonserver:21.02-py3 tritonserver \
  --model-store=${MODEL_REPOSITORY_PATH} \
  --exit-on-error=true \
  --model-control-mode=explicit
