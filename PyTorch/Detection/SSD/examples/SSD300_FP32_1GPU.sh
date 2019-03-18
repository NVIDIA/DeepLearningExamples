# This script launches SSD300 training in FP32 on 1 GPUs using 32 batch size
# Usage ./SSD300_FP32_1GPU.sh <path to this repository> <path to dataset> <additional flags>

python $1/main.py --backbone resnet50 --bs 32 --warmup 300 --data $2 ${@:3}
