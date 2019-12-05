ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.09-py3 

FROM tensorrtserver_client as trtis-client
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y python3
ARG version=6.0.1-1+cuda10.1
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb \
&& dpkg -i cuda-repo-*.deb \
&& wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb \
&& dpkg -i nvidia-machine-learning-repo-*.deb \
&& apt-get update \
&& apt-get install -y --no-install-recommends libnvinfer6=${version} libnvonnxparsers6=${version} libnvparsers6=${version} libnvinfer-plugin6=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python-libnvinfer=${version} python3-libnvinfer=${version}
RUN cp -r /usr/lib/python3.6/dist-packages/tensorrt /opt/conda/lib/python3.6/site-packages/tensorrt


ENV PATH=$PATH:/usr/src/tensorrt/bin
WORKDIR /tmp/onnx-trt
COPY trt/onnx-trt.patch .
RUN git clone https://github.com/onnx/onnx-tensorrt.git && cd onnx-tensorrt && git checkout  b677b9cbf19af803fa6f76d05ce558e657e4d8b6  && git submodule update --init --recursive && \
    patch -f < ../onnx-trt.patch && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DGPU_ARCHS="60 70 75" && make -j16 && make install && mv -f /usr/lib/libnvonnx* /usr/lib/x86_64-linux-gnu/ && ldconfig


# Here's a good place to install pip reqs from JoC repo.
# At the same step, also install TRT pip reqs
WORKDIR /tmp/pipReqs
COPY requirements.txt /tmp/pipReqs/pytRequirements.txt
COPY trt/requirements.txt /tmp/pipReqs/trtRequirements.txt
COPY trtis/requirements.txt /tmp/pipReqs/trtisRequirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends portaudio19-dev && pip install -r pytRequirements.txt && pip install -r trtRequirements.txt && pip install -r trtisRequirements.txt


#Copy the perf_client over
COPY --from=trtis-client /workspace/install/bin/perf_client /workspace/install/bin/perf_client
#Copy the python wheel and install with pip
COPY --from=trtis-client /workspace/install/python/tensorrtserver*.whl /tmp/
RUN pip install /tmp/tensorrtserver*.whl && rm /tmp/tensorrtserver*.whl

WORKDIR /workspace/jasper
COPY . .

