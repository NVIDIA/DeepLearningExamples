#!/bin/bash

# Constructs a docker image containing dependencies for execution of JASPER through TRT
echo "docker build . -f ./tensorrt/Dockerfile -t jasper:trt6"
docker build . -f ./tensorrt/Dockerfile -t jasper:trt6
