#!/bin/bash

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
EXTRA_MOUNTS=${3:-""}
IMAGE=${4:-"bert"}
DOCKER_BRIDGE=${5:-"host"}

docker run -it --rm \
  --gpus \"device=$NV_VISIBLE_DEVICES\" \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results \
  ${EXTRA_MOUNTS} \
  ${IMAGE} $CMD
