ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.06-tf1-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet
WORKDIR /workspace/unet

RUN pip install git+https://github.com/NVIDIA/dllogger
RUN pip install -r requirements.txt
