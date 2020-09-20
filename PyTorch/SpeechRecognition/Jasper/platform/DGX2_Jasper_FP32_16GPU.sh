#!/bin/bash

NUM_GPUS=16 BATCH_SIZE=64 GRADIENT_ACCUMULATION_STEPS=1 bash scripts/train.sh "$@"
