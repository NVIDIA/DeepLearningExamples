# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.10-py3
FROM ${FROM_IMAGE_NAME}

RUN apt update && apt install -y libsndfile1 && apt install -y sox && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/jasper

# Install requirements (do this first for better caching)
COPY requirements.txt .
RUN if [[ ! -z "$(command -v conda)" ]]; then conda install -y pyyaml==5.4.1; fi
RUN pip install --disable-pip-version-check -U -r requirements.txt

# Copy rest of files
COPY . .
