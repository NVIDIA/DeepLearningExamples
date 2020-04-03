#===============================================================================
#
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
#
# ==============================================================================

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.02-tf2-py3
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /workspace && mkdir -p /workspace
ADD . /workspace
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev python3-tk cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make sure python and pip points to pip3 and python3
RUN python -m pip install --upgrade pip && \
    pip --no-cache-dir --no-cache install \
        Cython \
        matplotlib \
        opencv-python-headless \
        mpi4py \
        Pillow \
        pytest \
        pyyaml && \
    git clone https://github.com/pybind/pybind11 /opt/pybind11 && \
    cd /opt/pybind11 && cmake . && make install && pip install . && \
    pip --no-cache-dir --no-cache install \
        'git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI' && \
    pip --no-cache-dir --no-cache install \
        'git+https://github.com/NVIDIA/dllogger'


# Update protobuf 3 to 3.3.0
RUN \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip && \
    unzip -u protoc-3.3.0-linux-x86_64.zip -d protoc3 && \
    mv protoc3/bin/* /usr/local/bin/ && \
    mv protoc3/include/* /usr/local/include/

