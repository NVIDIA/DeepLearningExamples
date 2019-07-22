ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:19.03-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2

RUN pip install toposort networkx pytest nltk tqdm html2text progressbar

WORKDIR /workspace
RUN git clone https://github.com/openai/gradient-checkpointing.git
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git

WORKDIR /workspace/bert
COPY . .

ENV PYTHONPATH=/workspace/bert
