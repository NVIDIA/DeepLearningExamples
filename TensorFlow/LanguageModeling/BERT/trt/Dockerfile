# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

FROM nvcr.io/nvidia/tensorrt:20.02-py3

LABEL maintainer="NVIDIA CORPORATION"

ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN chown trtuser /workspace

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    sudo \
    pbzip2 \
    pv \
    bzip2 \
    unzip

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_bat_linux.zip && unzip ngccli_bat_linux.zip && chmod u+x ngc && rm ngccli_bat_linux.zip ngc.md5 && echo "no-apikey\nascii\nno-org\nno-team\nno-ace\n" | ngc config set

# Install required Python packages
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Build with extra flags to enable plugins available on SM 70/75
RUN git clone https://github.com/NVIDIA/TensorRT.git
RUN cd TensorRT && git checkout 9b2d78a87e3d45eca22b6bfcb6e10777b932ef43 && git submodule update --init --recursive && mkdir build && cd build && cmake .. -DGPU_ARCHS="70 75" -DBUILD_PARSERS=OFF && make -j nvinfer_plugin
# Replace the default plugin library with the open-source plugins
RUN cp TensorRT/build/libnvinfer_plugin.so* /usr/lib/x86_64-linux-gnu/

WORKDIR /workspace/bert
USER trtuser
