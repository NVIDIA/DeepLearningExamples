#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#8 V100/A100 x 12 batch_per_gpu 

GPU=8
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
RESULTS='/results'
LOGFILE="$RESULTS/joblog.log"

if ! [ -d "$RESULTS" ]; then mkdir $RESULTS; fi

#Use a different argument with DATASET.TRAIN to use your own

python -m torch.distributed.launch --nproc_per_node=$GPU tools/train_net.py \
        --config-file $CONFIG \
        DTYPE "${DTYPE:-float16}" \
        NHWC "${NHWC:-True}" \
        DATALOADER.HYBRID "${HYBRID:-True}" \
        OUTPUT_DIR $RESULTS \
        | tee $LOGFILE
