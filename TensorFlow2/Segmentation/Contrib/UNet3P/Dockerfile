ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.12-tf2-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet3p
WORKDIR /workspace/unet3p

RUN pip install -r requirements.txt

#For opencv, inside docker run these commands
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# reinstall jupyterlab
RUN pip uninstall jupyterlab -y
RUN pip install jupyterlab
