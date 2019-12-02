FROM gitlab-master.nvidia.com:5005/dl/dgx/tensorflow:19.11-tf1-py3-devel

ADD . /workspace/vnet
WORKDIR /workspace/vnet

RUN pip install --upgrade pip
RUN pip install ./dllogger
RUN pip install --disable-pip-version-check -r requirements.txt


