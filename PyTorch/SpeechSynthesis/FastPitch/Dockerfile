ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.09-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /workspace/fastpitch
COPY . .
