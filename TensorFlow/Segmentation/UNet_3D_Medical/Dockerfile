ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.06-tf1-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

RUN pip install --upgrade pip
RUN pip install git+https://github.com/NVIDIA/dllogger
RUN pip install --disable-pip-version-check -r requirements.txt
