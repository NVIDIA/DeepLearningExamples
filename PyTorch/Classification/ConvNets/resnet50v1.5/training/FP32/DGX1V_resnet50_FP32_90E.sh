python ./launch.py --model resnet50 --precision FP32 --mode convergence --platform DGX1V /imagenet --epochs 90 --mixup 0.0 --workspace ${1:-./} --raport-file raport.json
