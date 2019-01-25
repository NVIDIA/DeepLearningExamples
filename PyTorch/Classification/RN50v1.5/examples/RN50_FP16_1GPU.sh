# This script launches ResNet50 training in FP16 on 1 GPUs using 256 batch size (256 per GPU)
# Usage ./RN50_FP16_1GPU.sh <path to this repository> <additional flags>

python $1/main.py -j5 -p 500 --arch resnet50 -c fanin --label-smoothing 0.1 -b 256 --lr 0.1 --epochs 90 --fp16 --static-loss-scale 256 $2 /data/imagenet
