# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
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

ARG TRITONSERVER_IMAGE=nvcr.io/nvidia/tritonserver:21.05-py3
ARG KALDI_IMAGE=nvcr.io/nvidia/kaldi:21.08-py3
ARG PYTHON_VER=3.8

#
# Kaldi shared library dependencies
#
FROM ${KALDI_IMAGE} as kaldi

#
# Builder image based on Triton Server SDK image
#
FROM ${TRITONSERVER_IMAGE}-sdk as builder
ARG PYTHON_VER

# Kaldi dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -yq --no-install-recommends \
        automake \
        autoconf \
        cmake \
        flac \
        gawk \
        libatlas3-base \
        libtool \
        python${PYTHON_VER} \
        python${PYTHON_VER}-dev \
        sox \
        subversion \
        unzip \
        bc \
        libatlas-base-dev \
        gfortran \
        zlib1g-dev; \
    rm -rf /var/lib/apt/lists/*

# Add Kaldi dependency
COPY --from=kaldi /opt/kaldi /opt/kaldi

# Set up Atlas
RUN set -eux; \
    ln -sf /usr/include/x86_64-linux-gnu/atlas     /usr/local/include/atlas; \
    ln -sf /usr/include/x86_64-linux-gnu/cblas.h   /usr/local/include/cblas.h; \
    ln -sf /usr/include/x86_64-linux-gnu/clapack.h /usr/local/include/clapack.h; \
    ln -sf /usr/lib/x86_64-linux-gnu/atlas         /usr/local/lib/atlas


#
# Kaldi custom backend build
#
FROM builder as backend-build

# Build the custom backend
COPY kaldi-asr-backend /workspace/triton-kaldi-backend
RUN set -eux; \
    cd /workspace/triton-kaldi-backend; \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
        -B build .; \
    cmake --build build --parallel; \
    cmake --install build


#
# Final server image
#
FROM ${TRITONSERVER_IMAGE}
ARG PYTHON_VER

# Kaldi dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -yq --no-install-recommends \
        automake \
        autoconf \
        cmake \
        flac \
        gawk \
        libatlas3-base \
        libtool \
        python${PYTHON_VER} \
        python${PYTHON_VER}-dev \
        sox \
        subversion \
        unzip \
        bc \
        libatlas-base-dev \
        zlib1g-dev; \
    rm -rf /var/lib/apt/lists/*

# Add Kaldi dependency
COPY --from=kaldi /opt/kaldi /opt/kaldi

# Add Kaldi custom backend shared library and scripts
COPY --from=backend-build /workspace/triton-kaldi-backend/install/backends/kaldi/libtriton_kaldi.so /workspace/model-repo/kaldi_online/1/
COPY scripts /workspace/scripts

# Setup entrypoint and environment
ENV LD_LIBRARY_PATH /opt/kaldi/src/lib/:/opt/tritonserver/lib:$LD_LIBRARY_PATH
COPY scripts/nvidia_kaldi_triton_entrypoint.sh /opt/triton-kaldi/
VOLUME /mnt/model-repo
ENTRYPOINT ["/opt/triton-kaldi/nvidia_kaldi_triton_entrypoint.sh"]
CMD ["tritonserver", "--model-repo=/workspace/model-repo"]
