# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.03-py3
FROM nvcr.io/nvidia/tritonserver:20.03-py3-clientsdk as trt
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install onnxruntime

COPY --from=trt /workspace/install /workspace/install/

ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}
RUN ls /workspace/install/python
RUN pip install /workspace/install/python/tensorrtserver-1.12.0-py3-none-linux_x86_64.whl

ENV PYTHONPATH /workspace/dlrm
WORKDIR /workspace/dlrm
COPY . .
