ARG FROM_IMAGE_NAME=nvcr.io/nvidia/mxnet:22.10-py3

FROM $FROM_IMAGE_NAME

WORKDIR /workspace/rn50

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV MXNET_CUDNN_AUTOTUNE_DEFAULT=0

COPY . .
