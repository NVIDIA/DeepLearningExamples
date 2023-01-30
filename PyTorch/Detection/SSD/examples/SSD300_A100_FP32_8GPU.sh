# This script launches SSD300 training in FP32 on 8 GPUs using 1024 batch size (128 per GPU)
# Usage ./SSD300_FP32_8GPU.sh <path to this repository> <path to dataset> <additional flags>

torchrun --nproc_per_node=8 $1/main.py --backbone resnet50 --learning-rate 2.7e-3 --warmup 1200 --bs 128 --no-amp --data $2 ${@:3}
