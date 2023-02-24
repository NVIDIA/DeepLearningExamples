# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

# run docker daemon with --default-runtime=nvidia for GPU detection during build
# multistage build for DGL with CUDA and FP16

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.01-py3

FROM ${FROM_IMAGE_NAME} AS dgl_builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y git build-essential python3-dev make cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /dgl
RUN git clone --branch 1.0.0 --recurse-submodules --depth 1 https://github.com/dmlc/dgl.git .
WORKDIR build
RUN export NCCL_ROOT=/usr \
    && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=ON -DCUDA_ARCH_BIN="60 70 80" -DCUDA_ARCH_PTX="80" \
        -DCUDA_ARCH_NAME="Manual" \
        -DUSE_FP16=ON \
        -DBUILD_TORCH=ON \
        -DUSE_NCCL=ON \
        -DUSE_SYSTEM_NCCL=ON \
        -DBUILD_WITH_SHARED_NCCL=ON \
        -DUSE_AVX=ON \
    && cmake --build .


FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/se3-transformer

# copy built DGL and install it
COPY --from=dgl_builder /dgl ./dgl
RUN cd dgl/python && python setup.py install && cd ../.. && rm -rf dgl

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD . .

ENV DGLBACKEND=pytorch
ENV OMP_NUM_THREADS=1


