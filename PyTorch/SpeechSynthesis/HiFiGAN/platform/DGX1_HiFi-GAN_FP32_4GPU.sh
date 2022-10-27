#!/bin/bash

set -a

: ${NUM_GPUS:=4}
: ${BATCH_SIZE:=32}
: ${GRAD_ACCUMULATION:=1}
: ${AMP:=false}

bash scripts/train_lj22khz.sh "$@"
