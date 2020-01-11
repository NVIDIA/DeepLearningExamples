#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
JASPER_REPO=${JASPER_REPO:-"${SCRIPT_DIR}/../../.."}

# Launch TRT JASPER container.

DATA_DIR=$1
CHECKPOINT_DIR=$2
RESULT_DIR=$3
PROGRAM_PATH=${PROGRAM_PATH}
    
if [ $# -lt 3 ]; then
    echo "Usage: ./launch.sh <DATA_DIR> <CHECKPOINT_DIR> <RESULT_DIR> (<SCRIPT_PATH>)"
    echo "All directory paths must be absolute paths and exist"
    exit 1
fi

for dir in $DATA_DIR $CHECKPOINT_DIR $RESULT_DIR; do
    if [[ $dir != /* ]]; then
        echo "All directory paths must be absolute paths!"
        echo "${dir} is not an absolute path"
        exit 1
    fi

    if [ ! -d $dir ]; then
        echo "All directory paths must exist!"
        echo "${dir} does not exist"
        exit 1
    fi
done


nvidia-docker run -it --rm \
  --runtime=nvidia \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $DATA_DIR:/datasets \
  -v $CHECKPOINT_DIR:/checkpoints/ \
  -v $RESULT_DIR:/results/ \
  -v ${JASPER_REPO}:/jasper \
  ${EXTRA_JASPER_ENV} \
  jasper:trt6 bash $PROGRAM_PATH
