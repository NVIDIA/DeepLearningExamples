#!/bin/bash
# ensure the TRTIS submodule is added and build the clients
SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/../../../
docker pull nvcr.io/nvidia/tensorrtserver:19.09-py3
git submodule update --init --recursive
docker build -t tensorrtserver_client -f ${PROJECT_DIR}/external/tensorrt-inference-server/Dockerfile.client ${PROJECT_DIR}/external/tensorrt-inference-server
docker build . --rm -f ${PROJECT_DIR}/trtis/Dockerfile -t jasper:trtis
