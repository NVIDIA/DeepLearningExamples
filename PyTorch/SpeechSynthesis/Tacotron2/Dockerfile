FROM nvcr.io/nvidia/pytorch:19.11-py3

ADD . /workspace/tacotron2
WORKDIR /workspace/tacotron2
RUN pip install -r requirements.txt
RUN pip --no-cache-dir --no-cache install  'git+https://github.com/NVIDIA/dllogger'
