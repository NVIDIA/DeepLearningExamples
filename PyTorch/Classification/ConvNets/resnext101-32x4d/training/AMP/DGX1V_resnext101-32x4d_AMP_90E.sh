python ./launch.py --model resnext101-32x4d --precision AMP --mode convergence --platform DGX1V /imagenet --epochs 90 --mixup 0.0 --workspace ${1:-./} --raport-file raport.json
