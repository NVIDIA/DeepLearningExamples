#!/bin/bash

DATA_DIR=${1:-"/mnt/dldata/bert"}
VOCAB_DIR=${2:-"/mnt/dldata/bert/vocab"}
CHECKPOINT_DIR=${3:-"/mnt/dldata/bert/pretrained_models_nvidia_pytorch"}

docker run -it --rm \
  --runtime=nvidia \
  -p 8888:8888 \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $DATA_DIR:/workspace/bert/data \
  -v $CHECKPOINT_DIR:/workspace/checkpoints \
  -v $VOCAB_DIR:/workspace/bert/vocab \
  -v $PWD/results:/results \
  bert bash
