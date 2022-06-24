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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.02-py3

######
# Tokenizers is only available pre-built on x86
#
FROM ${FROM_IMAGE_NAME} AS tokenizers_amd64
WORKDIR /wheelhouse
RUN pip download tokenizers==0.8.0

FROM quay.io/pypa/manylinux2014_aarch64 as tokenizers_arm64
ARG PYVER=38
RUN yum install -y openssl-devel
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly-2020-05-14 -y
ENV PATH="/root/.cargo/bin:$PATH"
ENV PYBIN=/opt/python/cp${PYVER}-cp${PYVER}/bin
ENV PYTHON_SYS_EXECUTABLE="$PYBIN/python"
RUN git clone -b python-v0.8.0 https://github.com/huggingface/tokenizers.git /opt/tokenizers
WORKDIR /opt/tokenizers/bindings/python
RUN "${PYBIN}/pip" install setuptools-rust \
 && "${PYBIN}/python" setup.py bdist_wheel \
 && rm -rf build/* \
 && for whl in dist/*.whl; do \
        auditwheel repair "$whl" -w dist/; \
    done \
 && rm dist/*-linux_* \
 && mkdir -p /wheelhouse \
 && mv dist/*.whl /wheelhouse

ARG TARGETARCH
FROM tokenizers_${TARGETARCH} AS tokenizers
#
#####


FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2

RUN --mount=from=tokenizers,source=/wheelhouse,target=/tmp/wheelhouse \
    pip install --no-cache-dir /tmp/wheelhouse/tokenizers*.whl

RUN pip install --no-cache-dir dataclasses gitpython rouge-score pynvml==8.0.4 \
  git+https://github.com/NVIDIA/dllogger pytorch-lightning==1.5.10 gdown sacrebleu

RUN pip install tqdm --upgrade

WORKDIR /workspace
RUN git clone https://github.com/artmatsak/cnn-dailymail.git
RUN git clone https://github.com/gcunhase/AMICorpusXML.git

WORKDIR /workspace/bart

COPY . .
