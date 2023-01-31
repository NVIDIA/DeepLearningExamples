# Copyright (c) 2020-2021 NVIDIA CORPORATION. All rights reserved.
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
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /workspace

WORKDIR /workspace/bert
RUN pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py nltk progressbar onnxruntime==1.12.0 tokenizers>=0.7\
 git+https://github.com/NVIDIA/dllogger wget

RUN apt-get install -y iputils-ping

COPY . .

# Install lddl
RUN apt-get install -y libjemalloc-dev
RUN pip install git+https://github.com/NVIDIA/lddl.git
RUN python -m nltk.downloader punkt

RUN pip install lamb_amp_opt/
