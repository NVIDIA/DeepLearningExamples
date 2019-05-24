#!/bin/bash
BASELINES=(87 330 569)
GPUS=(1 4 8)
PRECISION=FP32
TOLERANCE=0.11

for i in {1..4}
do
    GPU=${GPUS[$i]}
    
    MSG="Testing mixed precision training speed on $GPUS GPUs"
    CMD="bash ../../examples/SSD320_FP16_${GPU}GPU_BENCHMARK.sh /results/SSD320_FP16_${GPU}GPU ../../configs"

    if CMD=$CMD BASELINE=${BASELINES[$i]} TOLERANCE=$TOLERANCE MSG=$MSG bash ../../qa/testing_DGX1V_performance.sh
    then
        exit $?
    fi

done
