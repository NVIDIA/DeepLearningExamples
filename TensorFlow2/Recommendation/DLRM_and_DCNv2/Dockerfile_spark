# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

RUN apt update && \
    apt install -y openjdk-8-jdk && \
    apt install -y curl && \
    curl https://archive.apache.org/dist/spark/spark-3.0.1/spark-3.0.1-bin-hadoop3.2.tgz -o /opt/spark.tgz && \
    tar zxf /opt/spark.tgz -C /opt/ && \
    mv /opt/spark-3.0.1-bin-hadoop3.2 /opt/spark && \
    rm /opt/spark.tgz && \
    curl https://repo1.maven.org/maven2/ai/rapids/cudf/0.14/cudf-0.14-cuda10-2.jar -o /opt/cudf.jar && \
    curl https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/0.1.0/rapids-4-spark_2.12-0.1.0.jar -o /opt/rapids-4-spark.jar && \
    apt install -y git

ADD requirements.txt .
RUN apt install -y python3-pip && python3 -m pip install --upgrade pip && pip3 install -r requirements.txt

WORKDIR /workspace/dlrm

COPY . .

RUN mv /opt/cudf.jar  /opt/spark/jars && \
    mv /opt/rapids-4-spark.jar /opt/spark/jars/ && \
    mv /workspace/dlrm/preproc/gpu/get_gpu_resources.sh /opt/spark/conf/ && \
    mv /workspace/dlrm/preproc/gpu/spark-defaults.conf /opt/spark/conf/ && \
    rm -fr /workspace/dlrm/preproc/gpu

RUN chmod +x /opt/spark/conf/get_gpu_resources.sh
RUN /bin/bash -c "echo export PYSPARK_PYTHON=/usr/bin/python3 >> /etc/bash.bashrc; update-alternatives --install /usr/bin/python python /usr/bin/python3 10"

