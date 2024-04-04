# Copyright 2021-2024 NVIDIA CORPORATION

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SPDX-License-Identifier: Apache-2.0
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.12-py3

FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive
ENV DCGM_VERSION=2.2.9

ENV MODEL_NAVIGATOR_CONTAINER=1
ENV DGLBACKEND=pytorch
RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common curl python3-dev python3-pip python-is-python3 libb64-dev wget git wkhtmltopdf && \
    \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian buster stable" && \
    apt-get update && \
    apt-get install --no-install-recommends -y docker-ce docker-ce-cli containerd.io && \
    \
    . /etc/os-release && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey| apt-key add - && \
    curl -s -L "https://nvidia.github.io/nvidia-docker/${ID}${VERSION_ID}/nvidia-docker.list" > /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y nvidia-docker2 && \
    \
    curl -s -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    rm datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    apt-get install -y libb64-dev libb64-0d curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV PYTHONPATH /workspace

RUN rm /usr/lib/libxgboost.so

ADD requirements.txt /workspace/requirements.txt
ADD triton/requirements.txt /workspace/triton/requirements.txt
RUN pip install -r /workspace/requirements.txt
RUN pip install -r /workspace/triton/requirements.txt
RUN pip install nvidia-pyindex
RUN pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dgl==1.0.1 -f https://data.dgl.ai/wheels/cu117/repo.html

ADD ./hydra_plugins /workspace/hydra_plugins
RUN pip install /workspace/hydra_plugins/hydra_optuna_sweeper/
RUN pip install /workspace/hydra_plugins/hydra_joblib_launcher/
RUN pip install /workspace/hydra_plugins/hydra_multiprocessing_launcher/
RUN pip install /workspace/hydra_plugins/hydra_torchrun_launcher/
RUN cp /workspace/hydra_plugins/optuna_sweeper.py /usr/local/lib/python3.8/dist-packages/hydra/plugins/sweeper.py

ADD . /workspace
RUN rm -rf examples docker-examples tutorials
