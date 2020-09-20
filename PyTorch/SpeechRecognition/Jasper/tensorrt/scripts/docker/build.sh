#!/bin/bash

# Constructs a docker image containing dependencies for execution of JASPER through TensorRT
echo "docker build . -f ./tensorrt/Dockerfile -t jasper:tensorrt"
docker build . -f ./tensorrt/Dockerfile -t jasper:tensorrt
