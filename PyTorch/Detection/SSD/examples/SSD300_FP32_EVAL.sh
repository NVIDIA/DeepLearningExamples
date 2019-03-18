# This script evaluates SSD300 model in FP32 using 32 batch size on 1 GPU
# Usage: ./SSD300_FP32_EVAL.sh <path to this repository> <path to dataset> <path to checkpoint> <additional flags>

python $1/main.py --backbone resnet50 --ebs 32 --data $2 --mode evaluation --checkpoint $3 ${@:4}
