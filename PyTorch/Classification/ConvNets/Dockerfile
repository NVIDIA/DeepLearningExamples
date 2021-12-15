ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.03-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt /workspace/
WORKDIR /workspace/
RUN pip install nvidia-pyindex
RUN pip install nvidia-dllogger
RUN pip install wandb
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install -y dmidecode

ADD . /workspace/rn50
WORKDIR /workspace/rn50
