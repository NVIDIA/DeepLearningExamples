#!/usr/bin/env bash

# CLI args with defaults
BATCH_SIZE=${1:-240}
AMP=${2:-true}


# choices: 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
#          'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'
TASK=homo

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
  se3_transformer.runtime.inference \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --use_layer_norm \
  --norm \
  --load_ckpt_path model_qm9.pth \
  --task "$TASK"
