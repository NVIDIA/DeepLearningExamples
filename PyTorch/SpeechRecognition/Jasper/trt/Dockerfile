ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.10-py3 
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y python3

WORKDIR /tmp/onnx-trt
COPY trt/onnx-trt.patch .
RUN git clone https://github.com/onnx/onnx-tensorrt.git && cd onnx-tensorrt && git checkout 8716c9b && git submodule update --init --recursive && \
    patch -f < ../onnx-trt.patch && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DGPU_ARCHS="60 70 75" && make -j16 && make install && mv -f /usr/lib/libnvonnx* /usr/lib/x86_64-linux-gnu/ && ldconfig


# Here's a good place to install pip reqs from JoC repo.
# At the same step, also install TRT pip reqs
WORKDIR /tmp/pipReqs
COPY requirements.txt /tmp/pipReqs/jocRequirements.txt
COPY trt/requirements.txt /tmp/pipReqs/trtRequirements.txt
RUN pip install --disable-pip-version-check -U -r jocRequirements.txt -r trtRequirements.txt


WORKDIR /workspace/jasper
COPY . .

