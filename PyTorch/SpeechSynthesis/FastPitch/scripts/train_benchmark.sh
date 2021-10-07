#!/usr/bin/env bash

set -a

: ${AMP:=false}
: ${NUM_GPUS_SEQUENCE:="1 4 8"}
: ${EPOCHS:=30}
: ${OUTPUT_DIR:="./output"}
: ${BATCH_SIZE:=16}

for NUM_GPUS in $NUM_GPUS_SEQUENCE ; do
    GRAD_ACCUMULATION=$((256 / $BATCH_SIZE / $NUM_GPUS ))
    LOG_FILE=$OUTPUT_DIR/perf-train_amp-${AMP}_${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}.json
    bash scripts/train.sh "$@"
    rm -f $OUTPUT_DIR/FastPitch*.pt
done
