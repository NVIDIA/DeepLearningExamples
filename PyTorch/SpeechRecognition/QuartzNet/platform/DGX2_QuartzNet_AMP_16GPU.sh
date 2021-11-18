#!/bin/bash

set -a

: ${NUM_GPUS:=16}
: ${GPU_BATCH_SIZE:=36}
: ${GRAD_ACCUMULATION:=2}
: ${AMP=:true}

bash scripts/train.sh "$@"
