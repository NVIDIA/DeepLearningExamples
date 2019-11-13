#!/bin/bash

docker pull nvcr.io/nvidia/tensorrtserver:19.08-py3

#Will have to update submodule from root
git submodule update --init --recursive
cd tensorrt-inference-server && docker build -t tensorrtserver_client -f Dockerfile.client . && cd -

docker build . --rm -t bert
