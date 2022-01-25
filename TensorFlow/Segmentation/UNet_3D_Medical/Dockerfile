ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:21.10-tf1-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

RUN pip install git+https://github.com/NVIDIA/dllogger@v1.0.0#egg=dllogger
RUN pip install --disable-pip-version-check -r requirements.txt

ENV TF_GPU_HOST_MEM_LIMIT_IN_MB=120000
ENV XLA_FLAGS="--xla_multiheap_size_constraint_per_heap=2600000000"
ENV OMPI_MCA_coll_hcoll_enable=0