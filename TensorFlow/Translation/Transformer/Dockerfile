ARG FROM_IMAGE=nvcr.io/nvidia/tensorflow:19.03-py3
FROM ${FROM_IMAGE}

WORKDIR /research
ENV HOME /research

# Environment
ENV SEED 1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  curl \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Install Python 3 packages
RUN pip3 install \
  nvidia-ml-py \
  sacrebleu \
  numa 

# A patch for pynvml for Py2-to-Py3 transition
RUN sed -i 's/print c_count.value/print(c_count.value)/' /usr/local/lib/python3.5/dist-packages/pynvml.py

RUN apt-get update
RUN apt-get install -y cmake pkg-config libprotobuf9v5 protobuf-compiler libprotobuf-dev libgoogle-perftools-dev
RUN git clone https://github.com/google/sentencepiece.git /research/sentencepiece
RUN cd /research/sentencepiece \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j 8 \
  && make install \
  && ldconfig -v

WORKDIR /research/transformer
# Copy model into image - This can be overridden by mounting a volume to the same location.
COPY . .
ENV PYTHONPATH="/research/transformer/transformer:${PYTHONPATH}"
