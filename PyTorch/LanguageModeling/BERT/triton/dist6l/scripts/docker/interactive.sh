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

NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:=0}

docker run -it --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
  --net=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  -e WORKDIR="$(pwd)" \
  -e PYTHONPATH="$(pwd)" \
  -v "$(pwd)":"$(pwd)" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w "$(pwd)" \
  bert:latest bash
