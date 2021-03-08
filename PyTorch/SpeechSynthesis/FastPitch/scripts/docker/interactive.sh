#!/usr/bin/env bash

PORT=${PORT:-8888}

docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -p $PORT:$PORT -v $PWD:/workspace/fastpitch/ fastpitch:latest bash 
