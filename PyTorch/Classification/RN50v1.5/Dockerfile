FROM nvcr.io/nvidia/pytorch:19.05-py3

RUN git clone https://github.com/NVIDIA/apex \
        && cd apex \
        && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

ADD . /workspace/rn50
WORKDIR /workspace/rn50
