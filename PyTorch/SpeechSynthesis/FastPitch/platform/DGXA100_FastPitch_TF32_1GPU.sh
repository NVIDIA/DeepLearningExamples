#!/bin/bash

set -a

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=32}
: ${GRAD_ACCUMULATION:=8}
: ${AMP:=false}

bash scripts/train.sh "$@"
