#!/bin/bash

mkdir -p /tmp/results

python ./scripts/benchmarking/benchmark.py --mode inference --bench-warmup 100 --bench-iterations 200 --ngpus 1 --bs 1 2 4 8 16 32 64 128 --baseline ./scripts/benchmarking/baselines/DGX1V_RN50_tensorflow_infer_fp32.json --data_dir $1 --results_dir $2

python ./scripts/benchmarking/benchmark.py --mode inference --bench-warmup 100 --bench-iterations 200 --ngpus 1 --bs 1 2 4 8 16 32 64 96 --baseline ./scripts/benchmarking/baselines/DGX1V_RN50_tensorflow_infer_fp32.json --perf_args "use_xla" --data_dir $1 --results_dir $2/xla