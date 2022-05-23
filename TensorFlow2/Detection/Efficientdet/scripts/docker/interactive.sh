#!/bin/bash

docker run --runtime=nvidia \
-v $BACKBONE_CKPT:/workspace/checkpoints/efficientnet-b0-joc \
-v $CKPT:/workspace/checkpoints/efficientdet-tf2 \
-v ${DATA:-/mnt/nvdl/datasets/coco_master/coco2017_tfrecords}:/workspace/coco \
--rm --name=${name:-interactive} \
--shm-size=30g --ulimit memlock=-1 --ulimit stack=67108864 \
--ipc=host -p 0.0.0.0:${PORT:-6007}:${PORT:-6007} -t -i \
${DOCKER:-effdet_tf2:latest} bash