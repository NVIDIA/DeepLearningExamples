# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/merlin/merlin-tensorflow-training:22.03
FROM ${FROM_IMAGE_NAME}

ENV HOROVOD_CYCLE_TIME=0.1
ENV HOROVOD_FUSION_THRESHOLD=67108864
ENV HOROVOD_NUM_STREAMS=2

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV DCGM_VERSION=2.2.9

# Install perf_client required library
RUN apt-get update && \
    apt-get install -y libb64-dev libb64-0d curl && \
    curl -s -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    rm datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set workdir and python path
WORKDIR /workspace/wd2
ENV PYTHONPATH /workspace/wd2

# Install requirements
ADD requirements.txt requirements.txt
ADD triton/requirements.txt triton/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir -r triton/requirements.txt

# Add model files to workspace
COPY . .
