# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#


ARG DLFW_VERSION=23.02

FROM nvcr.io/nvidia/tensorflow:${DLFW_VERSION}-tf2-py3 as tf
FROM nvcr.io/nvidia/tritonserver:${DLFW_VERSION}-py3 as tritonserver

WORKDIR /hugectr_backend

# rapids components from the DLFW image
COPY --chown=1000:1000 --from=tf /usr/lib/libcudf* /usr/lib/
COPY --chown=1000:1000 --from=tf /usr/lib/libarrow* /usr/lib/
COPY --chown=1000:1000 --from=tf /usr/lib/libparquet* /usr/lib/
COPY --chown=1000:1000 --from=tf /usr/lib/cmake/arrow /usr/lib/cmake/arrow/
COPY --chown=1000:1000 --from=tf /usr/lib/libnvcomp* /usr/lib/
COPY --chown=1000:1000 --from=tf /usr/include/parquet /usr/include/parquet/
COPY --chown=1000:1000 --from=tf /usr/include/arrow /usr/include/arrow/
COPY --chown=1000:1000 --from=tf /usr/include/cudf /usr/include/cudf/
COPY --chown=1000:1000 --from=tf /usr/include/rmm /usr/include/rmm/
ARG PYTHON_VERSION=3.8
COPY --chown=1000:1000 --from=tf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/rmm /usr/local/lib/python${PYTHON_VERSION}/dist-packages/rmm
COPY --chown=1000:1000 --from=tf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/cuda /usr/local/lib/python${PYTHON_VERSION}/dist-packages/cuda
COPY --chown=1000:1000 --from=tf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/pyarrow /usr/local/lib/python${PYTHON_VERSION}/dist-packages/pyarrow
COPY --chown=1000:1000 --from=tf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/cudf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/cudf
COPY --chown=1000:1000 --from=tf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/dask_cudf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/dask_cudf
COPY --chown=1000:1000 --from=tf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/dask_cuda /usr/local/lib/python${PYTHON_VERSION}/dist-packages/dask_cuda

COPY --chown=1000:1000 --from=tf /usr/local/lib/python3.8/dist-packages/cudf-*.dist-info /usr/local/lib/python3.8/dist-packages/cudf.dist-info/
COPY --chown=1000:1000 --from=tf /usr/local/lib/python3.8/dist-packages/dask_cudf-*.dist-info /usr/local/lib/python3.8/dist-packages/dask_cudf.dist-info/
COPY --chown=1000:1000 --from=tf /usr/local/lib/python3.8/dist-packages/dask_cuda-*.dist-info /usr/local/lib/python3.8/dist-packages/dask_cuda.dist-info/
COPY --chown=1000:1000 --from=tf /usr/local/lib/python3.8/dist-packages/pyarrow-*.dist-info /usr/local/lib/python3.8/dist-packages/pyarrow.dist-info/
COPY --chown=1000:1000 --from=tf /usr/local/lib/python3.8/dist-packages/rmm-*.dist-info /usr/local/lib/python3.8/dist-packages/rmm.dist-info/


RUN apt update -y --fix-missing && \
    apt install -y --no-install-recommends software-properties-common && \
    apt install -y --no-install-recommends \
        ca-certificates \
        clang-format \
        curl \
        libcurl4-openssl-dev \
        git \
        graphviz \
        libarchive-dev \
        libb64-dev \
        libboost-serialization-dev \
        libexpat1-dev \
        libopenblas-dev \
        libre2-dev \
        libsasl2-2 \
        libssl-dev \
        libtbb-dev \
        openssl \
        policykit-1 \
        protobuf-compiler \
        python3 \
        python3-pip \
        python3-dev \
        rapidjson-dev \
        tree \
        wget \
        zlib1g-dev \
        # Required to build RocksDB and RdKafka..
        libgflags-dev \
        libbz2-dev \
        libsnappy-dev \
        liblz4-dev \
        libzstd-dev \
        libsasl2-dev \
        #   Required to build Protocol Buffers.
        autoconf automake libtool \
        #   Required to build Hadoop.
        default-jdk maven pkg-config \
        libpmem-dev \
        libsnappy-dev \
        #   Required to run Hadoop.
        openssh-server \
        # [ HugeCTR ]
        libaio-dev && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "cmake<3.25.0"

# Install spdlog
RUN git clone --branch v1.9.2 https://github.com/gabime/spdlog.git build-env && \
    pushd build-env && \
    mkdir build && cd build && cmake .. && make -j && make install && \
    popd && \
    rm -rf build-env

RUN git clone https://github.com/NVIDIA/HugeCTR.git &&\
    cd HugeCTR &&\
    git checkout 7134016 &&\
    git submodule update --init --recursive

RUN cd HugeCTR &&\
    mkdir -p build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80;90" -DENABLE_INFERENCE=ON .. &&\
    make -j && make install

RUN git clone https://github.com/triton-inference-server/hugectr_backend.git &&\
    cd hugectr_backend/hps_backend && git checkout release-23.02 && mkdir build && cd build &&\
    cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_COMMON_REPO_TAG=r23.02\
          -DTRITON_CORE_REPO_TAG=r23.02\
          -DTRITON_BACKEND_REPO_TAG=r23.02 ..

RUN cd hugectr_backend/hps_backend/build &&\
    export CPATH=/usr/local/hugectr/include:$CPATH && \
    export LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH && \
    make install && mkdir -p /opt/tritonserver/backends/hps &&\
    cp libtriton_hps.* /opt/tritonserver/backends/hps/

WORKDIR /opt/tritonserver
