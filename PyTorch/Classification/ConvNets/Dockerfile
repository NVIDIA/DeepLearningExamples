ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.12-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt /workspace/
WORKDIR /workspace/
RUN pip install --no-cache-dir -r requirements.txt
ADD . /workspace/rn50
WORKDIR /workspace/rn50
