ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.02-py3
ARG TRITON_CLIENT_IMAGE_NAME=nvcr.io/nvidia/tritonserver:21.02-py3-sdk
FROM ${TRITON_CLIENT_IMAGE_NAME} as triton-client
FROM ${FROM_IMAGE_NAME}

# Install Perf Client required library
RUN apt-get update && apt-get install -y libb64-dev libb64-0d

# Install Triton Client PythonAPI and copy Perf Client
COPY --from=triton-client /workspace/install/ /workspace/install/
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}
RUN find /workspace/install/python/ -iname triton*manylinux*.whl -exec pip install {}[all] \;

# Setup environment variables to access Triton Client binaries and libs
ENV PATH /workspace/install/bin:${PATH}
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}

ENV PYTHONPATH /workspace
WORKDIR /workspace

RUN pip install nvidia-pyindex
ADD requirements.txt /workspace/requirements.txt
ADD triton/requirements.txt /workspace/triton/requirements.txt
RUN pip install -r /workspace/requirements.txt
RUN pip install -r /workspace/triton/requirements.txt

ADD . /workspace

