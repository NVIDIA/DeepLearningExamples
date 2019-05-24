#!/bin/bash

mkdir -p /tmp/results

python ./scripts/benchmarking/benchmark.py --mode training --bench-warmup 200 --bench-iterations 500 --ngpus 1 4 8 16 --bs 64 128 256 --baseline ./scripts/benchmarking/baselines/DGX2_RN50_tensorflow_train_fp16.json --perf_args "use_tf_amp" --data_dir $1 --results_dir $2

python ./scripts/benchmarking/benchmark.py --mode training --bench-warmup 200 --bench-iterations 500 --ngpus 1 4 8 16 --bs 64 128 256 --baseline ./scripts/benchmarking/baselines/DGX2_RN50_tensorflow_train_fp16.json --perf_args "use_xla" "use_tf_amp" --data_dir $1 --results_dir $2/xla