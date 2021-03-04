python ./launch.py --model se-resnext101-32x4d --precision TF32 --mode convergence --platform DGXA100 /imagenet --workspace ${1:-./} --raport-file raport.json
