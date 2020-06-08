ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:19.12-tf1-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/transformer-xl/tf
RUN pip --no-cache-dir --no-cache install 'git+https://github.com/NVIDIA/dllogger'

ADD tf/ /workspace/transformer-xl/tf
