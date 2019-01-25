# This script launches ResNet50 training in FP32 on 1 GPUs using 128 batch size (128 per GPU)
# Usage ./RN50_FP32_1GPU.sh <path to this repository> <additional flags>

python $1/main.py -j5 -p 500 --arch resnet50 -c fanin --label-smoothing 0.1 -b 128 --lr 0.05 --epochs 90 $2 /data/imagenet
