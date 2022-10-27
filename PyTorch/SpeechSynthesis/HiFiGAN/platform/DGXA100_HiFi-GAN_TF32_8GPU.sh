#!/bin/bash

set -a

: ${NUM_GPUS:=8}
: ${BATCH_SIZE:=16}
: ${GRAD_ACCUMULATION:=1}
: ${AMP:=false}

bash scripts/train_lj22khz.sh "$@"
