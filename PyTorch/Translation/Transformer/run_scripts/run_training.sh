#! /bin/bash

nvidia-smi

python -m torch.distributed.launch --nproc_per_node 8 /workspace/translation/train.py \
  /data/wmt14_en_de_joined_dict \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006\
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 2560 \
  --seed 1 \
  --max-epoch 50 \
  --online-eval \
  --no-epoch-checkpoints \
  --no-progress-bar \
  --log-interval 1000 \
  --save-dir /results/checkpoints \
  --distributed-init-method env:// 
