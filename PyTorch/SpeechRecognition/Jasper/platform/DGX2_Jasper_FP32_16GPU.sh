#!/bin/bash

NUM_GPUS=16 BATCH_SIZE=64 GRAD_ACCUMULATION_STEPS=1 bash scripts/train.sh "$@"
