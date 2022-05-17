#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#8 GPUS x 12 batch/GPU

IMAGE=`docker build . --pull | tail -n 1 | awk '{print $3}'`
GPU=8
NAME='MRCNN_TRAIN'
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
PATH_TO_COCO='/home/sharath/Downloads/11419' #Location on COCO-2017 on local machine


#PATH_TO_RN50 - SCRIPT assumes R-50.pth exists in PATH_TO_COCO/models/R-50.pth

#Specify datasets of your choice with parameter DATASETS.TRAIN and DATASETS.TEST
MOUNT_LOCATION='/datasets/coco'
DOCKER_RESULTS='/results'
LOGFILE='joblog.log'
COMMAND="python -m torch.distributed.launch --nproc_per_node=$GPU tools/train_net.py \
        --config-file $CONFIG \
        DATASETS.TRAIN "(\"coco_2017_train\",)" \
        DATASETS.TEST "(\"coco_2017_val\",)" \
        SOLVER.BASE_LR 0.12 \
        SOLVER.MAX_ITER 16667 \
        SOLVER.STEPS \"(12000, 16000)\" \
        SOLVER.IMS_PER_BATCH 96 \
        DTYPE \"float16\" \
        OUTPUT_DIR $DOCKER_RESULTS \
        | tee $LOGFILE"

echo $COMMAND
docker run --runtime=nvidia -v $PATH_TO_COCO:/$MOUNT_LOCATION --rm --name=$NAME --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -t -i $IMAGE bash -c "$COMMAND"
