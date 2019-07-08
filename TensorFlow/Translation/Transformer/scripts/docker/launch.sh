#!/bin/bash

NV_GPU='0,1' nvidia-docker run -it --rm \
  --runtime=nvidia \
  -p 8888:8888 \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD:/research/transformer \
  -v $PWD/results:/results \
  transformer_tf bash
