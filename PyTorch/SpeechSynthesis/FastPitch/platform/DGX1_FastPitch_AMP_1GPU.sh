#!/bin/bash

set -a

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=16}
: ${GRAD_ACCUMULATION:=16}
: ${AMP:=true}

bash scripts/train.sh "$@"
