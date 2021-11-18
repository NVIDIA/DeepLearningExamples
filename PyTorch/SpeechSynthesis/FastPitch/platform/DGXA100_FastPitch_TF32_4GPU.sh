#!/bin/bash

set -a

: ${NUM_GPUS:=4}
: ${BATCH_SIZE:=32}
: ${GRAD_ACCUMULATION:=2}
: ${AMP:=false}

bash scripts/train.sh "$@"
