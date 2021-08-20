#!/usr/bin/env bash
# Script to benchmark multi-GPU training performance, with bases precomputation

# CLI args with defaults
BATCH_SIZE=${1:-240}
AMP=${2:-true}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
  se3_transformer.runtime.training \
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
