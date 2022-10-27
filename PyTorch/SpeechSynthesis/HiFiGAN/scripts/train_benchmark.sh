#!/usr/bin/env bash

export CUDNN_V8_API_ENABLED=1  # Keep the flag for older containers
export TORCH_CUDNN_V8_API_ENABLED=1

set -a

: ${RESUME:=false}
: ${AMP:=false}
: ${BATCH_SIZE:=16}
: ${NUM_GPUS:=8}  # 1 4 8
: ${OUTPUT_DIR:="./results/perf-train"}
: ${EPOCHS:=1000000}  # Prevents from saving a final checkpoint
: ${EPOCHS_THIS_JOB:=50}
: ${BMARK_EPOCHS_NUM:=40}

: ${VAL_INTERVAL:=100000}  # In num of epochs
: ${SAMPLES_INTERVAL:=100000}  # In num of epochs
: ${CHECKPOINT_INTERVAL:=100000}  # In num of epochs

GRAD_ACCUMULATION=$((128 / $BATCH_SIZE / $NUM_GPUS ))
LOG_FILE=$OUTPUT_DIR/perf-train_amp-${AMP}_${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}
LOG_FILE+=.json
bash scripts/train_lj22khz.sh "$@"
