ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.07-py3
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /opt
RUN rm -rf /opt/pytorch/apex ; \
  git clone https://github.com/NVIDIA/apex.git pytorch/apex ; \
  cd pytorch/apex ; \
  pip uninstall --yes apex; \
  git checkout 880ab925bce9f817a93988b021e12db5f67f7787;  \
  git pull; \
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

#WORKDIR /opt
#RUN cd pytorch/apex \
# && git fetch origin pull/334/head:multi_tensor_lamb_optimizer \
# && git checkout multi_tensor_lamb_optimizer \
# && python setup.py develop --cuda_ext --cpp_ext

WORKDIR /workspace
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git

WORKDIR /workspace/bert
RUN pip install tqdm boto3 requests six ipdb h5py html2text nltk progressbar
COPY . .
