#!/bin/bash

set -a

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=128}
: ${GRAD_ACCUMULATION:=1}
: ${AMP:=true}

bash scripts/train_lj22khz.sh "$@" --no_amp_grouped_conv
