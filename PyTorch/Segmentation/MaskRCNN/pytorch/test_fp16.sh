#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#Script for PyT CI
#CONFIG: 1x8x12

RESULTS_DIR='maskrcnn_coco2017_test'
REPO_DIR='/opt/pytorch/examples/Detectron_PyT/pytorch'
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'

LOGFILE=$REPO_DIR/results/$RESULTS_DIR/log_gpu_0_fp16.log
mkdir -p $REPO_DIR/results/$RESULTS_DIR

GPU=8
BBOX_THRESHOLD=0.375
MASK_THRESHOLD=0.341
THROUGHPUT=2.57
THRESHOLD=0.9

cd $REPO_DIR

python -m torch.distributed.launch --nproc_per_node=$GPU tools/train_net.py \
        --config-file $CONFIG \
        SOLVER.BASE_LR 0.12 \
        SOLVER.MAX_ITER 16667 \
        SOLVER.STEPS "(12000, 16000)" \
        SOLVER.IMS_PER_BATCH 96 \
        TEST.IMS_PER_BATCH 8 \
        DTYPE "float16" \
        OUTPUT_DIR results/$RESULTS_DIR \
        PATHS_CATALOG maskrcnn_benchmark/config/paths_catalog_ci.py \
        2>&1 | tee $LOGFILE

map=`cat $LOGFILE | grep -F 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' | tail -n 2 | awk -F' = ' '{print $2}' | egrep -o [0-9.]+`
bbox_map=`echo $map | awk -F' ' '{print $1}' | egrep -o [0-9.]+`
mask_map=`echo $map | awk -F' ' '{print $2}' | egrep -o [0-9.]+`
time=`cat $LOGFILE | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
throughput=$(echo $time 1.0 | awk '{ printf "%f", $2 / $1 }')

echo 'THRESHOLD:' $BBOX_THRESHOLD $MASK_THRESHOLD $THROUGHPUT
echo 'RESULT:' $map $throughput

ACCURACY_TEST_RESULT_BBOX=$(awk 'BEGIN {print ('${bbox_map}' >= '${BBOX_THRESHOLD}')}')
ACCURACY_TEST_RESULT_MASK=$(awk 'BEGIN {print ('${mask_map}' >= '${MASK_THRESHOLD}')}')

if [ $ACCURACY_TEST_RESULT_BBOX == 1 -a $ACCURACY_TEST_RESULT_MASK == 1 ];
    then
        echo "&&&& ACCURACY TEST PASSED"
    else
        echo "&&&& ACCURACY TEST FAILED"
    fi

PERFORMANCE_TEST_RESULT=$(awk 'BEGIN {print ('${throughput}' >= \
      ('${THROUGHPUT}' * '${THRESHOLD}'))}')

if [ $PERFORMANCE_TEST_RESULT == 1 ];
    then
        echo "&&&& PERFORMANCE TEST PASSED"
    else
        echo "&&&& PERFORMANCE TEST FAILED"
    fi
    
if [ $ACCURACY_TEST_RESULT_BBOX == 1 -a $ACCURACY_TEST_RESULT_MASK == 1 -a $PERFORMANCE_TEST_RESULT == 1 ];
    then
        echo "&&&& PASSED"
        exit 0
    else
        echo "&&&& FAILED"
        exit 1
    fi
