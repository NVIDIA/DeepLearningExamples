#!/bin/bash
# ensure the TRTIS submodule is added and build the clients
SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/../../../
docker pull nvcr.io/nvidia/tritonserver:20.10-py3-clientsdk
git submodule update --init --recursive
docker build . --rm -f ${PROJECT_DIR}/triton/Dockerfile -t jasper:triton
