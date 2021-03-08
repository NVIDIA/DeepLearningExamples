ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tritonserver:20.10-py3-clientsdk
FROM ${FROM_IMAGE_NAME}

RUN apt update && apt install -y python3-pyaudio libsndfile1

RUN pip3 install -U pip
RUN pip3 install onnxruntime unidecode inflect soundfile

WORKDIR /workspace/jasper
COPY . .
