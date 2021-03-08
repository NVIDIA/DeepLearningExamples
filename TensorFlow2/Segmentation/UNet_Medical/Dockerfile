ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:21.02-tf2-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet
WORKDIR /workspace/unet

RUN pip install --upgrade pip
RUN pip install git+https://github.com/NVIDIA/dllogger
RUN pip install -r requirements.txt
