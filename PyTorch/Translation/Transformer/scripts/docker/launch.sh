#!/bin/bash

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"0,1,2,3,4,5,6,7,8"}
DOCKER_BRIDGE=${3:-"host"}

nvidia-docker run -it --rm \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=${NV_VISIBLE_DEVICES} \
  -v $PWD/results:/results \
  -v $PWD/data:/data \
  transformer_pyt $CMD
