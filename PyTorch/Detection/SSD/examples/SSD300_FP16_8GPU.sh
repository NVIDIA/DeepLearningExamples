# This script launches SSD300 training in FP16 on 8 GPUs using 512 batch size (64 per GPU)
# Usage ./SSD300_FP16_8GPU.sh <path to this repository> <path to dataset> <additional flags>

torchrun --nproc_per_node=8 $1/main.py --backbone resnet50 --warmup 300 --bs 64 --data $2 ${@:3}
