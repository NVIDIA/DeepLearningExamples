# google automl efficientDet dockerfile

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.03-tf2-py3
FROM ${FROM_IMAGE_NAME}

# upgrade pip
RUN pip install --upgrade pip

# Copy detectron code and build
WORKDIR /workspace/effdet-tf2
COPY . .
RUN pip install -r requirements.txt

ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false tf_xla_async_io_level=0"
RUN pip install git+https://github.com/NVIDIA/dllogger
