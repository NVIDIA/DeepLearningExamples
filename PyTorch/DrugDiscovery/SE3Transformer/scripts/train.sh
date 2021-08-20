#!/usr/bin/env bash

# CLI args with defaults
BATCH_SIZE=${1:-240}
AMP=${2:-true}
NUM_EPOCHS=${3:-100}
LEARNING_RATE=${4:-0.002}
WEIGHT_DECAY=${5:-0.1}

# choices: 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
#          'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'
TASK=homo

python -m se3_transformer.runtime.training \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --use_layer_norm \
  --norm \
  --save_ckpt_path model_qm9.pth \
  --precompute_bases \
  --seed 42 \
  --task "$TASK"