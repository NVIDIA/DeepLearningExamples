# This script launches ResNet50 inference benchmark in FP32 on 1 GPU with 128 batch size

python ./main.py -j5 --arch resnet50 -b 128 --benchmark-inference /data/imagenet
