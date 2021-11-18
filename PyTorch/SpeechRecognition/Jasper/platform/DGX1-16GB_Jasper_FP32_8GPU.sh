#!/bin/bash

NUM_GPUS=8 BATCH_SIZE=64 GRAD_ACCUMULATION_STEPS=4 bash scripts/train.sh "$@"
