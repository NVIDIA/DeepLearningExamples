ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.11-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN pip install nvidia-pyindex
ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt
RUN pip install tensorflow-addons --upgrade

# AWS Client for data downloading
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip -qq awscliv2.zip
RUN ./aws/install
RUN rm -rf awscliv2.zip aws

ENV OMP_NUM_THREADS=2
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV OMPI_MCA_coll_hcoll_enable 0
ENV HCOLL_ENABLE_MCAST 0 

WORKDIR /workspace/nnunet_tf2
ADD . /workspace/nnunet_tf2
