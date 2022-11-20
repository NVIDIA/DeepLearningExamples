ARG FROM_IMAGE_NAME=nvcr.io/nvidia/paddlepaddle:22.08-py3

FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

ENV BERT_PREP_WORKING_DIR /workspace/bert/data
ADD requirements.txt /workspace/
WORKDIR /workspace/
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/attardi/wikiextractor.git && cd wikiextractor && git checkout 6408a430fc504a38b04d37ce5e7fc740191dee16 && cd ..
RUN git clone https://github.com/soskek/bookcorpus.git

ADD . /workspace/bert
WORKDIR /workspace/bert
