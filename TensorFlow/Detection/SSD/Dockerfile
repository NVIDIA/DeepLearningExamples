FROM nvcr.io/nvidia/tensorflow:19.03-py3 as base

FROM base as sha

RUN mkdir /sha
RUN cat `cat HEAD | cut -d' ' -f2` > /sha/repo_sha

FROM base as final

WORKDIR /workdir

RUN PROTOC_VERSION=3.0.0 && \
    PROTOC_ZIP=protoc-${PROTOC_VERSION}-linux-x86_64.zip && \
    curl -OL https://github.com/google/protobuf/releases/download/v$PROTOC_VERSION/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    rm -f $PROTOC_ZIP

COPY requirements.txt .
RUN pip install Cython
RUN pip install -r requirements.txt

WORKDIR models/research/
COPY models/research/ .
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH="/workdir/models/research/:/workdir/models/research/slim/:$PYTHONPATH"

COPY examples/ examples
COPY configs/ configs/
COPY download_all.sh download_all.sh

COPY --from=sha /sha .
