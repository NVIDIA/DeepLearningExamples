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
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:21.02-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 libcurl4 curl

WORKDIR /workspace
ENV HOME /workspace

WORKDIR /workspace
RUN git clone https://github.com/openai/gradient-checkpointing.git
RUN git clone https://github.com/attardi/wikiextractor.git && cd wikiextractor && git checkout 6408a430fc504a38b04d37ce5e7fc740191dee16 && cd ..
RUN git clone https://github.com/soskek/bookcorpus.git
RUN git clone https://github.com/titipata/pubmed_parser

RUN pip3 install /workspace/pubmed_parser

# Environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Python 3 packages
RUN pip3 install \
  requests \
  tqdm \
  horovod \
  sentencepiece \
  tensorflow_hub \
  pynvml \
  wget \
  progressbar \
  git+https://github.com/NVIDIA/dllogger

WORKDIR /workspace/bert_tf2
# Copy model into image - This can be overridden by mounting a volume to the same location.
COPY . .
ENV PYTHONPATH="/workspace/wikiextractor:/workspace/bert_tf2:${PYTHONPATH}"

#disable lazy compilatoin
ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false"

ENV TF_DEVICE_MIN_SYS_MEMORY_IN_MB=2048
