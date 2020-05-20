FROM nvcr.io/nvidia/tensorflow:20.02-tf2-py3

ADD . /workspace/unet
WORKDIR /workspace/unet

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
