#!/bin/bash

python ./qa/benchmark_performance.py --benchmark-mode inference --ngpus 1 --bs 2 4 8 16 32 --bench-warmup 100 --bench-iterations 200 --benchmark-file qa/benchmark_baselines/SSD300_pytorch_19.05_inference_fp32.json --data $1
