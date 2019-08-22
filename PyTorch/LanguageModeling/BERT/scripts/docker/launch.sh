#!/bin/bash

CHECKPOINT_DIR=${3:-"${PWD}/checkpoints"}
RESULTS_DIR=${4:-"${PWD}/results"}

docker run -it --rm \
  --runtime=nvidia \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ${PWD}:/workspace/bert \
  -v $CHECKPOINT_DIR:/workspace/checkpoints \
  -v $RESULTS_DIR:/results \
  bert_pyt bash
