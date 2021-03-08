ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.06-tf1-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/vnet
WORKDIR /workspace/vnet

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false"


