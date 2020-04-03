ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com:5005/dl/dgx/tensorflow:20.02-tf2-py3-devel
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet
WORKDIR /workspace/unet

RUN pip install --upgrade pip
RUN pip install -r requirements.txt