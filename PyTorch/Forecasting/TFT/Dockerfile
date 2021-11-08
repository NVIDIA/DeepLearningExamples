# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3

FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y libb64-dev libb64-0d
WORKDIR /workspace
#ENV PYTHONPATH /workspace
RUN pip uninstall -y typing

RUN apt update && apt install -y p7zip-full
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
RUN pip install --no-cache-dir -e git://github.com/NVIDIA/dllogger#egg=dllogger

COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/workspace"

# AMP monkey-patch
RUN sed -i 's/  def forward(ctx,/  @amp.custom_fwd\(cast_inputs=torch.float32\)\n  def forward(ctx,/g' /opt/conda/lib/python3.8/site-packages/apex/normalization/fused_layer_norm.py
RUN sed -i 's/  def backward(ctx,/  @amp.custom_bwd\n  def backward(ctx,/g' /opt/conda/lib/python3.8/site-packages/apex/normalization/fused_layer_norm.py
RUN sed -i 's/^import torch$/import torch\nfrom torch.cuda import amp/' /opt/conda/lib/python3.8/site-packages/apex/normalization/fused_layer_norm.py
