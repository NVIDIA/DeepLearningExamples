python ./multiproc.py --nproc_per_node 8 ./launch.py --model resnext101-32x4d --precision AMP --mode convergence --platform DGXA100 /imagenet --workspace ${1:-./} --raport-file raport.json
