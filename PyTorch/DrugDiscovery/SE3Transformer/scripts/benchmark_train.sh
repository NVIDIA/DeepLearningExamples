#!/usr/bin/env bash
# Script to benchmark single-GPU training performance, with bases precomputation

# CLI args with defaults
BATCH_SIZE=${1:-240}
AMP=${2:-true}

CUDA_VISIBLE_DEVICES=0 python -m se3_transformer.runtime.training \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --epochs 6 \
  --use_layer_norm \
  --norm \
  --save_ckpt_path model_qm9.pth \
  --task homo \
  --precompute_bases \
  --seed 42 \
  --benchmark
