# This script launches SSD300 training in FP32 on 4 GPUs using 128 batch size (32 per GPU)
# Usage ./SSD300_FP32_4GPU.sh <path to this repository> <path to dataset> <additional flags>

torchrun --nproc_per_node=4 $1/main.py --backbone resnet50 --warmup 300 --bs 32 --no-amp --data-layout channels_first --data $2 ${@:3}
