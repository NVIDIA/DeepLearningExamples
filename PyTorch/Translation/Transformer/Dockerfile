# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

FROM nvcr.io/nvidia/pytorch:19.05-py3 

# Install Python dependencies
RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir \
      sacrebleu \
      sentencepiece

RUN apt-get update
RUN apt-get install -y cmake pkg-config libprotobuf9v5 protobuf-compiler libprotobuf-dev libgoogle-perftools-dev
RUN git clone https://github.com/google/sentencepiece.git /workspace/sentencepiece
RUN cd /workspace/sentencepiece \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j 8 \
  && make install \
  && ldconfig -v

ENV PYTHONPATH=/workspace/translation/examples/translation/subword-nmt/
WORKDIR /workspace/translation
COPY . .
RUN git clone https://github.com/rsennrich/subword-nmt.git /workspace/translation/examples/translation/subword-nmt/
RUN pip install -e .
