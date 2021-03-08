#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
# limitations under the License.

CONTAINER_TF1x_BASE="nvcr.io/nvidia/tensorflow"
CONTAINER_TF1x_TAG="20.06-tf1-py3"

# ======================== Refresh base image ======================== #
docker pull "${CONTAINER_TF1x_BASE}:${CONTAINER_TF1x_TAG}"

# ========================== Build container ========================= #

echo -e "\n\nBuilding NVIDIA TF 1.x Container\n\n"

sleep 1

docker build -t joc_tensorflow_maskrcnn:tf1.x-py3 \
    --build-arg BASE_CONTAINER="${CONTAINER_TF1x_BASE}" \
    --build-arg IMG_TAG="${CONTAINER_TF1x_TAG}" \
    --build-arg FROM_IMAGE_NAME="nvcr.io/nvidia/tensorflow:20.06-tf1-py3" \
    .
