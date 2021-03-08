ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.06-tf1-py3
FROM ${FROM_IMAGE_NAME}



WORKDIR /workdir

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        libpmi2-0-dev \
 && rm -rf /var/lib/apt/lists/*

RUN PROTOC_VERSION=3.0.0 && \
    PROTOC_ZIP=protoc-${PROTOC_VERSION}-linux-x86_64.zip && \
    curl -OL https://github.com/google/protobuf/releases/download/v$PROTOC_VERSION/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    rm -f $PROTOC_ZIP

COPY requirements.txt .
RUN pip install Cython
RUN pip install -r requirements.txt
RUN pip --no-cache-dir --no-cache install 'git+https://github.com/NVIDIA/dllogger'

WORKDIR models/research/
COPY models/research/ .
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH="/workdir/models/research/:/workdir/models/research/slim/:$PYTHONPATH"

COPY examples/ examples
COPY configs/ configs/
COPY qa/ qa/
COPY download_all.sh download_all.sh
