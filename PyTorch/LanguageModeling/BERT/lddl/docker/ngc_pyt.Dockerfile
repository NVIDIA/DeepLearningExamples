ARG TAG
# Import a NGC PyTorch container as the base image.
# For more information on NGC PyTorch containers, please visit:
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:${TAG}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update -qq && \
    apt-get install -y git vim tmux && \
    rm -rf /var/cache/apk/*

RUN conda install -y jemalloc

# Copy the lddl source code to /workspace/lddl in the image, then install.
WORKDIR /workspace/lddl
ADD . .
RUN pip install ./

# Download the NLTK model data.
RUN python -m nltk.downloader punkt
