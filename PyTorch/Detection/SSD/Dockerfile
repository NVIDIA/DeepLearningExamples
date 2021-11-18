ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.05-py3
FROM ${FROM_IMAGE_NAME}

# Set working directory
WORKDIR /workspace/ssd

# Install nv-cocoapi
ENV COCOAPI_VERSION=2.0+nv0.6.0
RUN export COCOAPI_TAG=$(echo ${COCOAPI_VERSION} | sed 's/^.*+n//') \
 && pip install --no-cache-dir pybind11                             \
 && pip install --no-cache-dir git+https://github.com/NVIDIA/cocoapi.git@${COCOAPI_TAG}#subdirectory=PythonAPI
# Install dllogger
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/dllogger.git#egg=dllogger

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python3 -m pip install pycocotools==2.0.0

COPY . .
