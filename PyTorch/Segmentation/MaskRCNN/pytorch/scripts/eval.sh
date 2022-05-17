#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#Predictions will be stored in `FOLDER`/inference`
#1x8x4 DGX1V

GPU=8
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'

#This folder should a file called 'last_checkpoint' which contains the path to the actual checkpoint
FOLDER='/results'

#Example
#   /results
#      ------last_checkpoint
#      ------model.pth

#   
#  last_checkpoint
#-----------------------------
#|/results/model.pth         |  
#|                           |
#|                           |
#|                           |
#|                           |
#|                           |
#-----------------------------



LOGFILE="$FOLDER/joblog.log"

if ! [ -d "$FOLDER" ]; then mkdir $FOLDER; fi

python3 -m torch.distributed.launch --nproc_per_node=$GPU tools/test_net.py \
    --config-file $CONFIG \
    DATASETS.TEST "(\"coco_2017_val\",)" \
    DTYPE "float16" \
    OUTPUT_DIR $FOLDER \
    | tee $LOGFILE
