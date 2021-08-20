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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.07-py3

FROM ${FROM_IMAGE_NAME} AS dgl_builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y git build-essential python3-dev make cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /dgl
RUN git clone --branch v0.7.0 --recurse-submodules --depth 1 https://github.com/dmlc/dgl.git .
RUN sed -i 's/"35 50 60 70"/"60 70 80"/g' cmake/modules/CUDA.cmake
WORKDIR build
RUN cmake -DUSE_CUDA=ON -DUSE_FP16=ON ..
RUN make -j8


FROM ${FROM_IMAGE_NAME}

RUN rm -rf /workspace/*
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
