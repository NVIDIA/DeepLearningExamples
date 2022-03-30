#!/bin/bash

set -a

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=64}
: ${GRAD_ACCUMULATION:=2}
: ${AMP:=true}

bash scripts/train_lj22khz.sh "$@"
