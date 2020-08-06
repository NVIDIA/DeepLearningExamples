#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
JASPER_REPO=${JASPER_REPO:-"${SCRIPT_DIR}/../.."}

# Launch TRT JASPER container.

DATA_DIR=${1:-${DATA_DIR-"/datasets"}}
CHECKPOINT_DIR=${2:-${CHECKPOINT_DIR:-"/checkpoints"}}
RESULT_DIR=${3:-${RESULT_DIR:-"/results"}}
PROGRAM_PATH=${PROGRAM_PATH}

MOUNTS=""
MOUNTS+=" -v $DATA_DIR:/datasets"
MOUNTS+=" -v $CHECKPOINT_DIR:/checkpoints"
MOUNTS+=" -v $RESULT_DIR:/results"
MOUNTS+=" -v ${JASPER_REPO}:/jasper"

echo $MOUNTS
nvidia-docker run -it --rm \
  --runtime=nvidia \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${MOUNTS} \
  ${EXTRA_JASPER_ENV} \
  jasper:latest bash $PROGRAM_PATH
