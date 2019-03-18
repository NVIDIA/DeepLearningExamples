#!/bin/bash

docker run -it --rm \
  --runtime=nvidia \
  -p 8888:8888 \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results \
  bert bash

