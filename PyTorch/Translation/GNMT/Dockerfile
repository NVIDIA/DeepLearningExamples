FROM nvcr.io/nvidia/pytorch:19.01-py3

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ADD . /workspace/gnmt
WORKDIR /workspace/gnmt

RUN pip install -r requirements.txt
