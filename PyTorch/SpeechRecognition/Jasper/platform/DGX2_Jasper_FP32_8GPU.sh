#!/bin/bash

NUM_GPUS=8 AMP=true BATCH_SIZE=64 GRAD_ACCUMULATION_STEPS=2 bash scripts/train.sh "$@"
