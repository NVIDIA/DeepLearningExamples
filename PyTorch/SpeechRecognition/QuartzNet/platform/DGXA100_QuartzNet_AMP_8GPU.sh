#!/bin/bash

set -a

: ${NUM_GPUS:=8}
: ${GPU_BATCH_SIZE:=72}
: ${GRAD_ACCUMULATION:=2}
: ${AMP=:true}

bash scripts/train.sh "$@"
