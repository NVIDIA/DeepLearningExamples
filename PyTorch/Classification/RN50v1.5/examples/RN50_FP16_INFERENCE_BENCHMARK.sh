# This script launches ResNet50 inference benchmark in FP16 on 1 GPU with 256 batch size

python ./main.py -j5 --arch resnet50 -b 256 --fp16 --benchmark-inference /data/imagenet
