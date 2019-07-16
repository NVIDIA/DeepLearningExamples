#!/usr/bin/env bash

docker run --runtime=nvidia -v $PWD:/research/transformer \
    --rm --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 --ipc=host -t -i \
    transformer_tf bash -c "bash scripts/data_helper.sh"