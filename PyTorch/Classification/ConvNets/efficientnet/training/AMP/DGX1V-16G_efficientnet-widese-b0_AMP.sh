python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b0 --precision AMP --mode convergence --platform DGX1V-16G /imagenet --workspace ${1:-./} --raport-file raport.json
