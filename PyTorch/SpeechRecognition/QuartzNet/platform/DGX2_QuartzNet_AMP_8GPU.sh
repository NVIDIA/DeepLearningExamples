#!/bin/bash

set -a

: ${NUM_GPUS:=8}
: ${GPU_BATCH_SIZE:=36}
: ${GRAD_ACCUMULATION:=4}
: ${AMP=:true}

bash scripts/train.sh "$@"
