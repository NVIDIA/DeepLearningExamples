#!/bin/bash

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

nvidia-docker run --name zlj_bert_torch -it  \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 9002:9002 \
  -e LD_LIBRARY_PATH='/workspace/install/lib/' \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results \
  torch_bert_20.06-py3 $CMD
