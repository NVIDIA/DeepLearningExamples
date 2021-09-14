#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
QN_REPO=${QN_REPO:-"${SCRIPT_DIR}/../.."}

DATA_DIR=${1:-${DATA_DIR-${QN_REPO}"/datasets"}}
CHECKPOINT_DIR=${2:-${CHECKPOINT_DIR:-${QN_REPO}"/checkpoints"}}
RESULT_DIR=${3:-${RESULT_DIR:-${QN_REPO}"/results"}}
PROGRAM_PATH=${PROGRAM_PATH}

MOUNTS=""
MOUNTS+=" -v $DATA_DIR:/datasets"
MOUNTS+=" -v $CHECKPOINT_DIR:/checkpoints"
MOUNTS+=" -v $RESULT_DIR:/results"
MOUNTS+=" -v ${QN_REPO}:/quartznet"

docker run -it --rm --gpus all\
  --env PYTHONDONTWRITEBYTECODE=1 \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $MOUNTS \
  -w /quartznet \
  quartznet:latest bash $PROGRAM_PATH
