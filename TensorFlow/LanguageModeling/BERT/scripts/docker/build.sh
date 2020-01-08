#!/bin/bash

#docker pull nvcr.io/nvidia/tensorrtserver:19.08-py3
docker pull nvcr.io/nvidia/tensorrtserver:19.08-py3-clientsdk

#Will have to update submodule from root
git submodule update --init --recursive
cd tensorrt-inference-server && git checkout v1.5.0 && docker build -t tensorrtserver_client -f Dockerfile.client . && cd -

docker build --no-cache . --rm -t bert
