#!/bin/bash

CMD=${@:-/bin/bash}
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}
#IMAGE="nvcr.io/nvidia/tensorrtserver:19.08-py3"
IMAGE="bert"

nvidia-docker run --rm -it \
    --privileged \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
    -v $PWD:/workspace/bert \
    -v $PWD/results:/results \
    --workdir=/workspace/bert \
    $IMAGE $CMD
