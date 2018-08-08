#!/bin/bash

set -e

DATASET_DIR='data/wmt16_de_en'
RESULTS_DIR='gnmt_wmt16'

# run training
python3 -m multiproc train.py \
  --save ${RESULTS_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --seed 1 \
  --epochs 6 \
  --math fp16 \
  --print-freq 10 \
  --batch-size 128 \
  --model-config "{'num_layers': 4, 'hidden_size': 1024, 'dropout':0.2, 'share_embedding': True}" \
  --optimization-config "{'optimizer': 'Adam', 'lr': 5e-4}"
