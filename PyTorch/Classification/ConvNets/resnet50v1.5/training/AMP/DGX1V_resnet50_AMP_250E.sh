python ./launch.py --model resnet50 --precision AMP --mode convergence --platform DGX1V /imagenet --workspace ${1:-./} --raport-file raport.json
