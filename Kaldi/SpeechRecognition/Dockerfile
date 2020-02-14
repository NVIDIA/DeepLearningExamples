# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

FROM nvcr.io/nvidia/kaldi:19.12-online-beta-py3 as kb
ENV DEBIAN_FRONTEND=noninteractive

ARG PYVER=3.6

FROM nvcr.io/nvidia/tensorrtserver:19.12-py3

# Kaldi dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        automake \
        autoconf \
        cmake \
        flac \
        gawk \
        libatlas3-base \
        libtool \
        python$PYVER \
        python$PYVER-dev \
        sox \
        subversion \
        unzip \
        bc \
        libatlas-base-dev \
        zlib1g-dev

RUN mkdir /opt/trtis-kaldi && mkdir -p /workspace/model-repo/kaldi_online/1 && mkdir -p /mnt/model-repo
# Copying static files
COPY scripts /workspace/scripts

# Moving Kaldi to container
COPY --from=kb /opt/kaldi /opt/kaldi
ENV LD_LIBRARY_PATH /opt/kaldi/src/lib/:$LD_LIBRARY_PATH

# Building the custom backend
COPY trtis-kaldi-backend /workspace/trtis-kaldi-backend
#COPY --from=cbe /workspace/install/custom-backend-sdk /workspace/trtis-kaldi-backend/custom-backend-sdk
RUN cd /workspace/trtis-kaldi-backend && wget https://github.com/NVIDIA/tensorrt-inference-server/releases/download/v1.9.0/v1.9.0_ubuntu1804.custombackend.tar.gz -O custom-backend-sdk.tar.gz && tar -xzf custom-backend-sdk.tar.gz
RUN cd /workspace/trtis-kaldi-backend/ && make && cp libkaldi-trtisbackend.so /workspace/model-repo/kaldi_online/1/ && cd - && rm -r /workspace/trtis-kaldi-backend

COPY scripts/nvidia_kaldi_trtis_entrypoint.sh /opt/trtis-kaldi

ENTRYPOINT ["/opt/trtis-kaldi/nvidia_kaldi_trtis_entrypoint.sh"]
