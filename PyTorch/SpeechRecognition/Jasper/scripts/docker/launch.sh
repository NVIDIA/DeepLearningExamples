#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
: ${JASPER_REPO:="$SCRIPT_DIR/../.."}

: ${DATA_DIR:=${1:-"$JASPER_REPO/datasets"}}
: ${CHECKPOINT_DIR:=${2:-"$JASPER_REPO/checkpoints"}}
: ${OUTPUT_DIR:=${3:-"$JASPER_REPO/results"}}
: ${SCRIPT:=${4:-}}

mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $OUTPUT_DIR

MOUNTS=""
MOUNTS+=" -v $DATA_DIR:/datasets"
MOUNTS+=" -v $CHECKPOINT_DIR:/checkpoints"
MOUNTS+=" -v $OUTPUT_DIR:/results"
MOUNTS+=" -v $JASPER_REPO:/workspace/jasper"

echo $MOUNTS
docker run -it --rm --gpus all \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $MOUNTS \
  $EXTRA_JASPER_ENV \
  -w /workspace/jasper \
  jasper:latest bash $SCRIPT
