# This script launches SSD300 training in FP16 on 1 GPUs using 64 batch size
# Usage bash SSD300_FP16_1GPU.sh <path to this repository> <path to dataset> <additional flags>

python $1/main.py --backbone resnet50 --warmup 300 --bs 64 --amp --data $2 ${@:3}
