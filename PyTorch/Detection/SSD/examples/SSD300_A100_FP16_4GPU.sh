# This script launches SSD300 training in FP16 on 4 GPUs using 1024 batch size (256 per GPU)
# Usage ./SSD300_FP16_4GPU.sh <path to this repository> <path to dataset> <additional flags>

python -m torch.distributed.launch --nproc_per_node=4 $1/main.py --backbone resnet50 --learning-rate 2.7e-3 --warmup 1200 --bs 256 --amp --data $2 ${@:3}
