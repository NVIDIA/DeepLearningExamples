#!/bin/bash

MOUNTS=$1
CMD=${2:-"bash"}
IMAGE=${3:-"lddl"}
GPUS=${4:-"all"}

docker run \
  --gpus \"device=${GPUS}\" \
  --init \
  -it \
  --rm \
  --network=host \
  --ipc=host \
  -v $PWD:/workspace/lddl \
  ${MOUNTS} \
  ${IMAGE} \
  ${CMD}
