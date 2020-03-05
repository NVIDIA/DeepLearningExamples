ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.01-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/tacotron2
WORKDIR /workspace/tacotron2
RUN pip install --no-cache-dir -r requirements.txt
