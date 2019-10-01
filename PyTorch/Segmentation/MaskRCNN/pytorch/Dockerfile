# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.05-py3
FROM ${FROM_IMAGE_NAME}

# Install Python dependencies
RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir \
      mlperf-compliance==0.0.10 \
      opencv-python==3.4.1.15 \
      yacs

WORKDIR /opt

# Copy detectron code and build
WORKDIR /workspace/object_detection
RUN mkdir -p /datasets/data
RUN mkdir /results
COPY . .
RUN pip install -e .

ENV OMP_NUM_THREADS=1
