#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#Predictions will be stored in `FOLDER`/inference`
#1x8x4 DGX1V

GPU=1
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
DTYPE=$1 

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
    --skip-eval \
    DATASETS.TEST "(\"coco_2017_val\",)" \
    DTYPE "$DTYPE" \
    NHWC "${NHWC:-True}" \
    DATALOADER.HYBRID "${HYBRID:-True}" \
    OUTPUT_DIR $FOLDER \
    TEST.IMS_PER_BATCH $2 \
    | tee $LOGFILE

#2019-02-22 00:05:39,954 maskrcnn_benchmark.inference INFO: Total inference time: 0:04:55.840343 (0.05916806864738464 s / img per device, on 1 devices)

time=`cat $LOGFILE | grep -F 'maskrcnn_benchmark.inference INFO: Total inference time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
calc=$(echo $time 1.0 | awk '{ printf "%f", $2 / $1 }')
echo "Inference perf is: "$calc" FPS"
