#!/bin/bash

nvidia-docker run -it --rm --shm-size=16g -v $PWD:/workspace/fastspeech/ fastspeech bash
# nvidia-docker run -it -u $(id -u):$(id -g) --rm --shm-size=16g -v $PWD:/workspace/fastspeech/ fastspeech bash
# --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host
