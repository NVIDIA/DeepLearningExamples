#!/bin/bash

function get_dataloader_workers {
    gpus=$(nvidia-smi -i 0 --query-gpu=count --format=csv,noheader)
    core=$(nproc --all)
    workers=$((core/gpus-2))
    workers=$((workers>16?16:workers))
    echo ${workers}
}
WORKERS=$(get_dataloader_workers)

./distributed_train.sh 8 /workspace/object_detection/datasets/waymo --model efficientdet_d0 -b 8 --amp --lr 0.2 --sync-bn --opt fusedmomentum --warmup-epochs 1 --output /model --worker $WORKERS --fill-color mean --model-ema --model-ema-decay 0.999 --eval-after 24 --epochs 24 --save-checkpoint-interval 1 --smoothing 0.0 --waymo --remove-weights class_net box_net anchor --input_size 1536 --num_classes 3 --resume --freeze-layers backbone --waymo-train /workspace/object_detection/datasets/waymo/training/images --waymo-val /workspace/object_detection/datasets/waymo/validation/images --waymo-val-annotation /waymo/validation/annotations/annotations-subset.json --waymo-train-annotation /waymo/training/annotations/annotations.json --initial-checkpoint /checkpoints/model_best.pth.tar