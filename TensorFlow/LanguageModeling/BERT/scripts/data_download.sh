#!/usr/bin/env bash

docker run --runtime=nvidia -v $PWD:/workspace/bert \
    --rm --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 --ipc=host -t -i \
    bert bash -c "bash scripts/data_download_helper.sh"