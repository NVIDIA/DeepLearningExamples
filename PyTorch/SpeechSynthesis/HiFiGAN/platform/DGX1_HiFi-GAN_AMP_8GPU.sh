#!/bin/bash

set -a

: ${NUM_GPUS:=8}
: ${BATCH_SIZE:=16}
: ${GRAD_ACCUMULATION:=1}
: ${AMP:=true}

bash scripts/train_lj22khz.sh "$@"
