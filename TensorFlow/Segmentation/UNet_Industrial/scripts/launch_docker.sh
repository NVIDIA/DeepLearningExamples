#!/usr/bin/env bash

DATASET_DIR=$(realpath -s $1)
RESULT_DIR=$(realpath -s $2)

if [[ ! -e ${DATASET_DIR} ]]; then
    echo "creating ${DATASET_DIR} ..."
    mkdir -p "${DATASET_DIR}"
fi

if [[ ! -e ${RESULT_DIR} ]]; then
    echo "creating ${RESULT_DIR} ..."
    mkdir -p "${RESULT_DIR}"
fi

# Build the docker container
docker build . --rm -t unet_industrial:latest

# start the container with nvidia-docker
nvidia-docker run -it --rm \
    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${DATASET_DIR}:/data/dagm2007/ \
    -v ${RESULT_DIR}:/results \
    unet_industrial:latest