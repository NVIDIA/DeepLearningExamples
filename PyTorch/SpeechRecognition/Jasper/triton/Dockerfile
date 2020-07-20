ARG FROM_IMAGE_NAME=nvcr.io/nvidian/pytorch:20.03-py3
ARG TRITON_BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.03.1-py3-clientsdk
FROM ${TRITON_BASE_IMAGE} as triton
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install onnxruntime

ADD triton/requirements.txt .
RUN pip install -r requirements.txt

ADD tensorrt/requirements.txt .
RUN pip install -r requirements.txt

COPY --from=triton /opt/tritonserver/qa/pkgs/tensorrtserver-1.13.0-py3-none-linux_x86_64.whl ./tensorrtserver-1.13.0-py3-none-linux_x86_64.whl

RUN pip install tensorrtserver-1.13.0-py3-none-linux_x86_64.whl

WORKDIR /workspace/jasper
COPY . .
RUN pip install --no-cache-dir -e .
