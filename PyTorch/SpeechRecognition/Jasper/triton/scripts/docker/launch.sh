#!/bin/bash
# Launch TRT JASPER container.

SCRIPT_DIR=$(cd $(dirname $0); pwd)
JASPER_REPO=${JASPER_REPO:-"${SCRIPT_DIR}/../../.."}


DATA_DIR=${DATA_DIR:-"/datasets"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/checkpoints"}
RESULT_DIR=${RESULT_DIR:-"/results"}
PROGRAM_PATH=${PROGRAM_PATH}
    

MOUNTS=""
if [ ! -z "$DATA_DIR" ]; 
then
    MOUNTS="$MOUNTS -v $DATA_DIR:/datasets "
fi

if [ ! -z "$CHECKPOINT_DIR" ]; 
then
    MOUNTS="$MOUNTS -v $CHECKPOINT_DIR:/checkpoints "
fi

if [ ! -z "$RESULT_DIR" ]; 
then
    MOUNTS="$MOUNTS -v $RESULT_DIR:/results "
fi

echo $MOUNTS
docker run -it --rm \
  --gpus=all \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${MOUNTS} \
  -v ${JASPER_REPO}:/jasper \
  ${EXTRA_JASPER_ENV} \
  jasper:triton bash $PROGRAM_PATH
