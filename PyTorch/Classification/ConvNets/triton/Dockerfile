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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.07-py3
ARG TRITON_BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.07-py3-clientsdk

FROM ${TRITON_BASE_IMAGE} as trt
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install onnxruntime

COPY --from=trt /workspace/v2.1.0.clients.tar.gz ./v2.1.0.clients.tar.gz
COPY --from=trt /workspace/install/bin/perf_client /bin/perf_client

RUN tar -xzf v2.1.0.clients.tar.gz \
    && pip install ./python/tritonclientutils-2.1.0-py3-none-any.whl


RUN apt update && apt install -y libb64-0d

WORKDIR /workspace/rn50
COPY . .

