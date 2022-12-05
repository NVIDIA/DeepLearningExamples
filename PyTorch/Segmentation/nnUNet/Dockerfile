ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.11-py3
FROM ${FROM_IMAGE_NAME} 

ADD ./requirements.txt .
RUN pip install --disable-pip-version-check -r requirements.txt
RUN pip install monai==1.0.0 --no-dependencies
RUN pip install numpy --upgrade

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip -qq awscliv2.zip
RUN ./aws/install
RUN rm -rf awscliv2.zip aws

ENV OMP_NUM_THREADS=2
WORKDIR /workspace/nnunet_pyt
ADD . /workspace/nnunet_pyt 
RUN cp utils/instance_norm.py /usr/local/lib/python3.8/dist-packages/apex/normalization