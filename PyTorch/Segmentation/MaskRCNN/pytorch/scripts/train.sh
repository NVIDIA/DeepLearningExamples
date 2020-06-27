#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#8 V100 x 4 batch_per_gpu DGX1V

GPU=8
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
RESULTS='/results'
LOGFILE="$RESULTS/joblog.log"

if ! [ -d "$RESULTS" ]; then mkdir $RESULTS; fi

#Use a different argument with DATASET.TRAIN to use your own

python -m torch.distributed.launch --nproc_per_node=$GPU tools/train_net.py \
        --config-file $CONFIG \
        DTYPE "float16" \
        OUTPUT_DIR $RESULTS \
        | tee $LOGFILE
