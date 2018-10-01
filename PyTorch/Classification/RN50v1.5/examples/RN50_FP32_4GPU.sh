# This script launches ResNet50 training in FP32 on 4 GPUs using 512 batch size (128 per GPU)
# Usage ./RN50_FP32_4GPU.sh <path to this repository> <additional flags>

python -m apex.parallel.multiproc $1/main.py -j5 -p 500 --arch resnet50 -b 128 --lr 0.2 --warmup 5 --epochs 90 $2 /data/imagenet
