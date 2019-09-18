
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.09-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y python3
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb \
&& dpkg -i cuda-repo-*.deb \
&& wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb \
&& dpkg -i nvidia-machine-learning-repo-*.deb \
&& apt-get update \
&& apt-get install -y --no-install-recommends python-libnvinfer python3-libnvinfer


RUN cp -r /usr/lib/python3.6/dist-packages/tensorrt /opt/conda/lib/python3.6/site-packages/tensorrt
# Add TensorRT executable to path (trtexec)
ENV PATH=$PATH:/usr/src/tensorrt/bin


# Here's a good place to install pip reqs from JoC repo.
# At the same step, also install TRT pip reqs
WORKDIR /tmp/pipReqs
COPY requirements.txt /tmp/pipReqs/jocRequirements.txt
COPY trt/requirements.txt /tmp/pipReqs/trtRequirements.txt
RUN pip install --disable-pip-version-check -U -r jocRequirements.txt -r trtRequirements.txt

# These packages are required for running preprocessing on the dataset to acquire manifest files and the like
RUN apt-get install -y libsndfile1 && apt-get install -y ffmpeg sox && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/jasper
COPY . .

