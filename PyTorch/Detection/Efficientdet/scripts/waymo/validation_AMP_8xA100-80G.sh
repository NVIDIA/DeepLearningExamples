#!/bin/bash
NUM_PROC=$1
rm -rf *.json

python -u -m bind_launch --nproc_per_node=${NUM_PROC} validate.py '/workspace/object_detection/datasets/waymo' --model efficientdet_d0 -b 10 --amp --waymo --use-ema --input_size 1536 --num_classes 3 --waymo-val /waymo/validation/images --waymo-val-annotation /waymo/validation/annotations/annotations.json --checkpoint /model/checkpoint.pth.tar --torchscript
