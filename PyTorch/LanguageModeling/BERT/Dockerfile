ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.05-py3-devel
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract


#WORKDIR /opt
#RUN cd pytorch/apex \
# && git fetch origin pull/182/head:norm_fix \
# && git checkout norm_fix \
# && python setup.py develop --cuda_ext --cpp_ext


WORKDIR /opt
RUN cd pytorch/apex ; \
  pip uninstall apex; \
  pip uninstall apex; \ 
  git checkout master;  \
  git pull; \
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /workspace
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git

WORKDIR /workspace/bert
COPY . .
RUN pip install tqdm boto3 requests six ipdb h5py html2text nltk progressbar