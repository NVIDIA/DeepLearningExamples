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

FROM nvcr.io/nvidia/tritonserver:20.06-v1-py3-clientsdk AS trtserver
FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract mc iputils-ping wget

WORKDIR /workspace/speech_ai_demo/

# Copy the perf_client over
COPY --from=trtserver /workspace/install/ /workspace/install/
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}

# set up env variables
ENV PATH="$PATH:/opt/conda/bin"
RUN cd /workspace/speech_ai_demo/

# jupyter lab extensions
RUN conda install -c conda-forge jupyterlab=1.0 ipywidgets=7.5 nodejs=10.13 python-sounddevice librosa unidecode inflect
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN pip install /workspace/install/python/tensorrtserver*.whl

# Copy the python wheel and install with pip
COPY --from=trtserver /workspace/install/python/tensorrtserver*.whl /tmp/
RUN pip install /tmp/tensorrtserver*.whl && rm /tmp/tensorrtserver*.whl

COPY start_jupyter.sh /workspace/speech_ai_demo/
COPY speech_ai_demo/utils /workspace/speech_ai_demo/utils
COPY speech_ai_demo/speech_ai_demo.ipynb /workspace/speech_ai_demo/
RUN chmod a+x /workspace/speech_ai_demo/start_jupyter.sh
