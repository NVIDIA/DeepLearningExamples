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
#
# author: Tomasz Grel (tgrel@nvidia.com)

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:23.06-tf2-py3
FROM nvcr.io/nvidia/tritonserver:23.06-py3-sdk as clientsdk
FROM ${FROM_IMAGE_NAME} as base

ARG DISTRIBUTED_EMBEDDINGS_COMMIT=45cffaa8

WORKDIR /dlrm

RUN apt update && apt install -y libb64-dev tree

ADD requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN rm -rf distributed-embeddings &&\
    git clone https://github.com/NVIDIA-Merlin/distributed-embeddings.git &&\
    cd distributed-embeddings &&\
    git checkout ${DISTRIBUTED_EMBEDDINGS_COMMIT} &&\
    git submodule init && git submodule update &&\
    pip uninstall -y distributed-embeddings &&\
    make clean &&\
    make pip_pkg -j all &&\
    pip install artifacts/*.whl &&\
    cd ..

ADD tensorflow-dot-based-interact tensorflow-dot-based-interact
RUN mkdir -p /usr/local/lib/python3.8/dist-packages/tensorflow/include/third_party/gpus/cuda/ &&\
    cd tensorflow-dot-based-interact &&\
    make clean &&\
    pip uninstall -y tensorflow-dot-based-interact &&\
    make &&\
    make pkg &&\
    pip install ./artifacts/tensorflow_dot_based_interact-*.whl &&\
    cd ..

COPY --from=clientsdk /workspace/install/python/tritonclient-2.35.0-py3-*.whl /dlrm/
RUN if [[ "$(uname -m)" == "x86_64" ]]; \
    then echo x86; pip install tritonclient-2.35.0-py3-none-manylinux1_x86_64.whl[all]; \
    else echo arm; pip install tritonclient-2.35.0-py3-none-manylinux2014_aarch64.whl[all]; \
    fi

ADD . .

ENV HOROVOD_CYCLE_TIME=0.2
ENV HOROVOD_ENABLE_ASYNC_COMPLETION=1
