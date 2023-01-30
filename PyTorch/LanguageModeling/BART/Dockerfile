# syntax = docker/dockerfile:1
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
# limitations under the License.
# ==============================================================================

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.08-py3
FROM ${FROM_IMAGE_NAME}


RUN apt-get update
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
RUN git clone https://github.com/abisee/cnn-dailymail.git
RUN git clone https://github.com/gcunhase/AMICorpusXML.git

# Re-build apex
RUN git clone https://github.com/nv-joseli/apex.git
RUN cd apex && \
    git checkout bf16lamb && \
    NVCC_APPEND_FLAGS='--threads 1' pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /workspace/bart

COPY . .
