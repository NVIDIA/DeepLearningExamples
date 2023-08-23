ARG FROM_IMAGE_NAME=nvcr.io/nvidia/paddlepaddle:23.06-py3
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /workspace/

WORKDIR /workspace/bert
RUN pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py nltk progressbar tokenizers>=0.7\
 git+https://github.com/NVIDIA/dllogger wget

RUN apt-get install -y iputils-ping

COPY . .

RUN apt-get install -y libjemalloc-dev
RUN pip install git+https://github.com/NVIDIA/lddl.git
RUN python -m nltk.downloader punkt
