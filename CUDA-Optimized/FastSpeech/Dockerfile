ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.03-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/fastspeech
WORKDIR /workspace/fastspeech

RUN sh ./scripts/install.sh