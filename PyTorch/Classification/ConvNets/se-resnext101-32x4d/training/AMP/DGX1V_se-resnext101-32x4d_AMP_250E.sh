python ./launch.py --model se-resnext101-32x4d --precision AMP --mode convergence --platform DGX1V /imagenet --workspace ${1:-./} --raport-file raport.json
