#!/bin/bash

function get_dataloader_workers {
    gpus=$(nvidia-smi -i 0 --query-gpu=count --format=csv,noheader)
    core=$(nproc --all)
    workers=$((core/gpus-2))
    workers=$((workers>16?16:workers))
    echo ${workers}
}
WORKERS=$(get_dataloader_workers)

./distributed_train.sh 8 /workspace/object_detection/datasets/coco --model efficientdet_d0 -b 60 --lr 0.65 --amp --opt fusedmomentum --warmup-epochs 20 --lr-noise 0.4 0.9 --output /model --worker ${WORKERS} --fill-color mean --model-ema --model-ema-decay 0.999 --eval-after 200 --epochs 300 --resume --smoothing 0.0 --pretrained-backbone-path /backbone_checkpoints/jocbackbone_statedict_B0.pth --memory-format nchw --sync-bn --fused-focal-loss --seed 12711
