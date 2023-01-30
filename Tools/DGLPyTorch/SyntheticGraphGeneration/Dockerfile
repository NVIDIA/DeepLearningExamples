# - image
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.12-py3
FROM ${FROM_IMAGE_NAME}

# - SNAP dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y gnuplot graphviz \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=interactive

# - requirements
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
RUN pip install -U --no-deps dython

# - dir
WORKDIR /workspace/
COPY . .

# - envs
RUN echo 'alias syngen="python syngen"' >> ~/.bashrc
ENV PYTHONPATH "${PYTHONPATH}:/workspace/"
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
