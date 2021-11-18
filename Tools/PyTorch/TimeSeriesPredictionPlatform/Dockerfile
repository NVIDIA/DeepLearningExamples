#SPDX-License-Identifier: Apache-2.0
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.09-py3

FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive
ENV DCGM_VERSION=2.2.9

ENV MODEL_NAVIGATOR_CONTAINER=1
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


# Install perf_client required library
RUN apt-get update && \
    apt-get install -y libb64-dev libb64-0d curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set workdir and python path
WORKDIR /workspace
ENV PYTHONPATH /workspace

RUN pip install --upgrade pip
ADD requirements.txt /workspace/requirements.txt
ADD triton/requirements.txt /workspace/triton/requirements.txt
RUN pip install -r /workspace/requirements.txt
RUN pip install -r /workspace/triton/requirements.txt
RUN pip install nvidia-pyindex
RUN pip install nvidia-dllogger
RUN pip install --no-cache-dir -r requirements.txt -f https://data.dgl.ai/wheels/repo.html

# Add model files to workspace
ADD . /workspace


# AMP monkey-patch
RUN sed -i 's/  def forward(ctx,/  @amp.custom_fwd\(cast_inputs=torch.float32\)\n  def forward(ctx,/g' /opt/conda/lib/python3.8/site-packages/apex/normalization/fused_layer_norm.py
RUN sed -i 's/  def backward(ctx,/  @amp.custom_bwd\n  def backward(ctx,/g' /opt/conda/lib/python3.8/site-packages/apex/normalization/fused_layer_norm.py
RUN sed -i 's/^import torch$/import torch\nfrom torch.cuda import amp/' /opt/conda/lib/python3.8/site-packages/apex/normalization/fused_layer_norm.py
RUN rm -rf examples
RUN rm -rf docker-examples
RUN rm -rf tutorial
