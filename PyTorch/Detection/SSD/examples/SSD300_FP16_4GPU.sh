# This script launches SSD300 training in FP16 on 4 GPUs using 256 batch size (64 per GPU)
# Usage ./SSD300_FP16_4GPU.sh <path to this repository> <path to dataset> <additional flags>

python -m torch.distributed.launch --nproc_per_node=4 $1/main.py --backbone resnet50 --warmup 300 --bs 64 --amp --data $2 ${@:3}
