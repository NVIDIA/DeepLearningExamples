#!/bin/bash

set -a

: ${NUM_GPUS:=8}
: ${BATCH_SIZE:=16}
: ${GRAD_ACCUMULATION:=2}
: ${AMP:=true}

bash scripts/train.sh "$@"
