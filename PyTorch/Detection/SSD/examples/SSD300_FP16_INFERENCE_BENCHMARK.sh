# This script launches SSD300 inference benchmark in FP16 on 1 GPU with 64 batch size
# Usage bash SSD300_FP16_INFERENCE_BENCHMARK.sh <path to this repository> <path to dataset> <additional flags>

python $1/main.py --backbone resnet50 --mode benchmark-inference --bs 64 --amp --data $2 ${@:3}
