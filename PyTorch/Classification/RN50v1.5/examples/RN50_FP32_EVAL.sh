# This script evaluates ResNet50 model in FP32 using 64 batch size on 1 GPU
# Usage: ./RN50_FP32_EVAL.sh <path to this repository> <path to checkpoint>

python $1/main.py -j5 p 100 --arch resnet50 -b 128 --resume $2 --evaluate /data/imagenet
