#!/bin/bash

DATA_DIR=${1:-"/data/tfrecords"}
DALI_DIR=${2}

BATCH_SIZE_TO_TEST="1 2 4 8 16 32 64 128"
INFERENCE_BENCHMARK=$(mktemp /tmp/inference-benchmark.XXXXXX)

function test_configuration() {
    echo "Testing configuration: $1" | tee -a $INFERENCE_BENCHMARK

    for BATCH in $BATCH_SIZE_TO_TEST; do
        python ./main.py --arch=resnext101-32x4d --mode=inference_benchmark --warmup_steps 50 --num_iter 400 --iter_unit batch \
            --batch_size $BATCH --data_dir=$DATA_DIR --results_dir=/tmp/results $2 | tail -n2 | head -n1 | sed \
            's/^DLL \([0-9]*-\)*[0-9]* \([0-9]*:\)*[0-9]*.[0-9]* - ()/Results for BS='$BATCH'/' | tee -a $INFERENCE_BENCHMARK

        if [ ! $? -eq 0 ]; then
            echo "Failed test on batch size $BATCH_SIZE"
            exit 1
        fi
    done
}

test_configuration "FP32 nodali noxla"
test_configuration "FP32 nodali xla" "--xla"
test_configuration "FP16 nodali noxla" "--amp"
test_configuration "FP16 nodali xla" "--amp --xla"

if [ ! -z $DALI_DIR ]; then
    test_configuration "FP16 dali xla" "--amp --xla --dali --data_idx_dir ${DALI_DIR}"
fi

cat $INFERENCE_BENCHMARK
rm $INFERENCE_BENCHMARK
