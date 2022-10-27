#!/bin/bash

set -a

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=128}
: ${GRAD_ACCUMULATION:=1}
: ${AMP:=false}

bash scripts/train_lj22khz.sh "$@"
