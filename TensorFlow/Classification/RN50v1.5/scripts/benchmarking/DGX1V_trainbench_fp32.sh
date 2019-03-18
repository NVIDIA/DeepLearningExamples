#!/bin/bash

mkdir -p /tmp/results

python ./scripts/benchmarking/benchmark.py --mode training --bench-warmup 200 --bench-iterations 500 --ngpus 1 4 8 --bs 32 64 128 --baseline ./scripts/benchmarking/baselines/RN50_tensorflow_train_fp32.json --data_dir $1 --results_dir $2