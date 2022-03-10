# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.10-py3
FROM ${FROM_IMAGE_NAME}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV DCGM_VERSION=2.0.13

# Install perf_client required library
RUN apt-get update && \
    apt-get install -y libb64-dev libb64-0d curl pbzip2 pv bzip2 cabextract wget libb64-dev libb64-0d && \
    curl -s -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    rm datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /workspace
RUN git clone https://github.com/attardi/wikiextractor.git && cd wikiextractor && git checkout 6408a430fc504a38b04d37ce5e7fc740191dee16 && cd ..
RUN git clone https://github.com/soskek/bookcorpus.git

# Setup environment variables to access Triton Client binaries and libs
ENV PATH /workspace/install/bin:${PATH}
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH /workspace/bert

WORKDIR /workspace/bert
ADD requirements.txt /workspace/bert/requirements.txt
ADD triton/requirements.txt /workspace/bert/triton/requirements.txt
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir -r /workspace/bert/requirements.txt
RUN pip install --no-cache-dir -r /workspace/bert/triton/requirements.txt

COPY . .
