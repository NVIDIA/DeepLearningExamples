#!/usr/bin/env bash

docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -v $PWD:/workspace/hifigan/ hifigan:latest bash 
