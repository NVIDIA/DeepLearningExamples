FROM nvcr.io/nvidia/pytorch:19.03-py3

ADD . /workspace/tacotron2
WORKDIR /workspace/tacotron2
RUN pip install -r requirements.txt
RUN cd /workspace; \
    git clone https://github.com/NVIDIA/apex.git; \
    cd /workspace/apex; \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /workspace/tacotron2
