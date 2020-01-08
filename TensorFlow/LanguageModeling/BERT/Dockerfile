ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:19.08-py3

FROM tensorrtserver_client as trt

FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 libcurl4 libcurl4-openssl-dev

RUN pip install toposort networkx pytest nltk tqdm html2text progressbar

WORKDIR /workspace
RUN git clone https://github.com/openai/gradient-checkpointing.git
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git
RUN git clone https://github.com/titipata/pubmed_parser

RUN pip3 install /workspace/pubmed_parser

#Copy the perf_client over
COPY --from=trt /workspace/install/ /workspace/install/

#Install the python wheel with pip
RUN pip install /workspace/install/python/tensorrtserver*.whl

WORKDIR /workspace/bert
COPY . .

ENV PYTHONPATH /workspace/bert
ENV BERT_PREP_WORKING_DIR /workspace/bert/data
ENV PATH //workspace/install/bin:${PATH}
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}
