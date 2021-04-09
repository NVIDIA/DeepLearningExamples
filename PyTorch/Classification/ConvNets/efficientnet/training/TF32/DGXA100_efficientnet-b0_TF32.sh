python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-b0 --precision TF32 --mode convergence --platform DGXA100 /imagenet --workspace ${1:-./} --raport-file raport.json
