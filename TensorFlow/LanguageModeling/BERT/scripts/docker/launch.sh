#!/bin/bash

CMD=${@:-/bin/bash}
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}


nvidia-docker run --name=zlj_bert_tf -it \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
    -p 9001:9001 \
    -v $PWD:/workspace/bert \
    -v $PWD/results:/results \
    tf_bert_20.06-py3 $CMD
