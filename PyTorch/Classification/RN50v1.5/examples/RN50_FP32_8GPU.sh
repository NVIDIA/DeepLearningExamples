# This script launches ResNet50 training in FP32 on 8 GPUs using 1024 batch size (128 per GPU)
# Usage ./RN50_FP32_8GPU.sh <path to this repository> <additional flags>

python $1/multiproc.py --nproc_per_node 8 $1/main.py -j5 -p 500 --arch resnet50 -c fanin --label-smoothing 0.1 -b 128 --lr 0.4 --warmup 5 --epochs 90 $2 /data/imagenet
