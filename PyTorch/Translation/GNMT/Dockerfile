FROM nvcr.io/nvidia/pytorch:19.05-py3

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /workspace/gnmt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD . /workspace/gnmt
