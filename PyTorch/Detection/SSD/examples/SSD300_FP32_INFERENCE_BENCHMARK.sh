# This script launches SSD300 inference benchmark in FP32 on 1 GPU with 64 batch size
# Usage bash SSD300_FP32_INFERENCE_BENCHMARK.sh <path to this repository> <path to dataset> <additional flags>

python $1/main.py --backbone resnet50 --warmup 300 --mode benchmark-inference --bs 32 --data $2 ${@:3}
