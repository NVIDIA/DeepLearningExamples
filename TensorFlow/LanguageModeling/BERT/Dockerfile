ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.06-tf1-py3

FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 libcurl4 curl libb64-dev
RUN pip install toposort networkx pytest nltk tqdm progressbar pynvml
RUN pip --no-cache-dir --no-cache install git+https://github.com/NVIDIA/dllogger wget

WORKDIR /workspace
RUN git clone https://github.com/openai/gradient-checkpointing.git
RUN git clone https://github.com/attardi/wikiextractor.git && cd wikiextractor && git checkout 6408a430fc504a38b04d37ce5e7fc740191dee16 && cd ..
RUN git clone https://github.com/soskek/bookcorpus.git
RUN git clone https://github.com/titipata/pubmed_parser


RUN pip3 install /workspace/pubmed_parser

#Install tritonclient
RUN pip install tritonclient[all]==2.5.0

WORKDIR /workspace/bert
COPY . .

ENV PYTHONPATH /workspace/bert
ENV BERT_PREP_WORKING_DIR /workspace/bert/data
ENV PATH //workspace/install/bin:${PATH}
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}
