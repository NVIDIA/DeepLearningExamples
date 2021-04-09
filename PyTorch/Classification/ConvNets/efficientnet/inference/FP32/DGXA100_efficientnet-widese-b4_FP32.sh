
python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b4 --precision FP32 --mode benchmark_inference --platform DGXA100 /imagenet -b 1 --workspace ${1:-./} --raport-file raport_1.json
python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b4 --precision FP32 --mode benchmark_inference --platform DGXA100 /imagenet -b 2 --workspace ${1:-./} --raport-file raport_2.json
python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b4 --precision FP32 --mode benchmark_inference --platform DGXA100 /imagenet -b 4 --workspace ${1:-./} --raport-file raport_4.json
python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b4 --precision FP32 --mode benchmark_inference --platform DGXA100 /imagenet -b 8 --workspace ${1:-./} --raport-file raport_8.json
python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b4 --precision FP32 --mode benchmark_inference --platform DGXA100 /imagenet -b 16 --workspace ${1:-./} --raport-file raport_16.json
python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b4 --precision FP32 --mode benchmark_inference --platform DGXA100 /imagenet -b 32 --workspace ${1:-./} --raport-file raport_32.json
python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b4 --precision FP32 --mode benchmark_inference --platform DGXA100 /imagenet -b 64 --workspace ${1:-./} --raport-file raport_64.json
python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-widese-b4 --precision FP32 --mode benchmark_inference --platform DGXA100 /imagenet -b 128 --workspace ${1:-./} --raport-file raport_128.json
