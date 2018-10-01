# This script launches ResNet50 training in FP16 on 4 GPUs using 1024 batch size (256 per GPU)
# Usage ./RN50_FP16_4GPU.sh <path to this repository> <additional flags>

python -m apex.parallel.multiproc $1/main.py -j5 -p 500 --arch resnet50 -b 256 --lr 0.4 --warmup 5 --epochs 90 --fp16 --static-loss-scale 256 $2 /data/imagenet
