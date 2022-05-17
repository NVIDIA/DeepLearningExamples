#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#Predictions will be stored in `FOLDER`/inference`
#1x8x4 DGX1V

GPU=1

# uncomment below to use default 
# CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
CONFIG="$1" 

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

#Use a different argument with DATASET.TEST to use your own

python3 -m torch.distributed.launch --nproc_per_node=$GPU tools/test_net.py \
    --config-file $CONFIG \
    --skip-eval \
    DTYPE "float16" \
    DATASETS.TEST "(\"coco_2017_val\",)" \
    OUTPUT_DIR $FOLDER \
    TEST.IMS_PER_BATCH 1 \
    | tee $LOGFILE
