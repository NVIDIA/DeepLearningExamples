#!/bin/bash
rm -rf *.json

python validate.py '/workspace/object_detection/datasets/coco/' --model efficientdet_d4 -b ${BATCH_SIZE:-8} --torchscript --use-ema --amp --checkpoint ${CKPT_PATH:-/checkpoints/Effdet_B0_test.pth}
