python ./launch.py --model resnet50 --precision AMP --mode convergence --platform DGX2V /imagenet --epochs 90 --mixup 0.0 --workspace ${1:-./} --raport-file raport.json
