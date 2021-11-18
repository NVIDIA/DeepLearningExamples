#!/usr/bin/env bash
# Script to benchmark inference performance, without bases precomputation

# CLI args with defaults
BATCH_SIZE=${1:-240}
AMP=${2:-true}

CUDA_VISIBLE_DEVICES=0 python -m se3_transformer.runtime.inference \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --use_layer_norm \
  --norm \
  --task homo \
  --seed 42 \
  --benchmark
