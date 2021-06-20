ARG FROM_IMAGE_NAME=nvcr.io/nvidia/mxnet:20.12-py3

FROM $FROM_IMAGE_NAME

WORKDIR /workspace/rn50

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
