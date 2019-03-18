#!/bin/bash

mkdir -p /tmp/results

python ./scripts/benchmarking/benchmark.py --mode inference --use_tf_amp --bench-warmup 100 --bench-iterations 200 --ngpus 1 --bs 1 2 4 8 16 32 64 128 256 --baseline ./scripts/benchmarking/baselines/RN50_tensorflow_infer_fp16.json  --data_dir $1 --results_dir $2
