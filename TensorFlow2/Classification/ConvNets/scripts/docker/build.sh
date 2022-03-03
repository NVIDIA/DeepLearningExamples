#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
set -euxo pipefail
arg1=$1
# CONTAINER_TF2x_BASE="gitlab-master.nvidia.com:5005/dl/dgx/tensorflow"
# CONTAINER_TF2x_TAG="21.10-tf2-py3-devel"
CONTAINER_TF2x_BASE="nvcr.io/nvidia/tensorflow"
CONTAINER_TF2x_TAG="21.09-tf2-py3"
# ======================== Refresh base image ======================== #
docker pull "${CONTAINER_TF2x_BASE}:${CONTAINER_TF2x_TAG}"
# ========================== Build container ========================= #
echo -e "\n\nBuilding Effnet_SavedModel Container\n\n"
echo $arg1
sleep 1
# the image name is given by the user ($1). Example: nvcr.io/nvidian/efficientnet-tf2:v2-ga-tf2-py3
docker build -t "$arg1" \
    --build-arg FROM_IMAGE_NAME="${CONTAINER_TF2x_BASE}:${CONTAINER_TF2x_TAG}" \
    .
