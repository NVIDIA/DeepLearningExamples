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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3

FROM ${FROM_IMAGE_NAME}

ENV IBV_DRIVERS /usr/lib/libibverbs/libmlx5

RUN pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger.git#egg=dllogger
RUN pip install --upgrade --pre omegaconf
RUN pip install --upgrade --pre tabulate

WORKDIR /workspace/object_detection
ENV PYTHONPATH "${PYTHONPATH}:/workspace/object_detection"
COPY . .
RUN export FORCE_CUDA=1 && pip install -v .
RUN mkdir -p /datasets/data
RUN mkdir /results
