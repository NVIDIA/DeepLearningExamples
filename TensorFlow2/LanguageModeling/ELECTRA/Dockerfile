# syntax = docker/dockerfile:1
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.07-tf2-py3

######
# Tokenizers is only available pre-built on x86
#
FROM ${FROM_IMAGE_NAME} AS tokenizers_amd64
WORKDIR /wheelhouse
RUN pip download tokenizers==0.7.0

FROM quay.io/pypa/manylinux2014_aarch64 as tokenizers_arm64
ARG PYVER=38
RUN yum install -y openssl-devel
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly-2019-11-01 -y
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
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

RUN --mount=from=tokenizers,source=/wheelhouse,target=/tmp/wheelhouse \
    pip install --no-cache-dir /tmp/wheelhouse/tokenizers*.whl

ENV DATA_PREP_WORKING_DIR /workspace/electra/data
WORKDIR /workspace
RUN git clone https://github.com/attardi/wikiextractor.git && cd wikiextractor && git checkout 6408a430fc504a38b04d37ce5e7fc740191dee16 && cd ..
RUN git clone https://github.com/soskek/bookcorpus.git

WORKDIR /workspace/electra

RUN pip install --no-cache-dir tqdm boto3 requests six ipdb h5py nltk progressbar filelock  \
 git+https://github.com/NVIDIA/dllogger \
 nvidia-ml-py3==7.352.0

RUN apt-get install -y iputils-ping
COPY . .
