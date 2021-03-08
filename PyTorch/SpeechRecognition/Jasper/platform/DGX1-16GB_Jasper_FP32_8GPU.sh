#!/bin/bash

NUM_GPUS=8 BATCH_SIZE=64 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/train.sh "$@"
