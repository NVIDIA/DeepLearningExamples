#!/bin/bash
rm -rf *.json

python -u -m bind_launch --nproc_per_node=${NUM_PROC:-1} validate.py '/workspace/object_detection/datasets/coco/' --model efficientdet_d0 -b ${BATCH_SIZE:-8} --torchscript --use-ema --checkpoint ${CKPT_PATH:-/checkpoints/Effdet_B0.pth}