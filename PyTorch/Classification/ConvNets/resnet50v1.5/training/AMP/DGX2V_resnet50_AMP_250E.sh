python ./multiproc.py --nproc_per_node 8 ./launch.py --model resnet50 --precision AMP --mode convergence --platform DGX2V /imagenet --workspace ${1:-./} --raport-file raport.json
