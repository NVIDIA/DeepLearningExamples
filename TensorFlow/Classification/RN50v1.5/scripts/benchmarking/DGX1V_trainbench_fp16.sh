#!/bin/bash

mkdir -p /tmp/results

python ./scripts/benchmarking/benchmark.py --mode training --use_tf_amp --bench-warmup 200 --bench-iterations 500 --ngpus 1 4 8 --bs 64 128 256 --baseline ./scripts/benchmarking/baselines/RN50_tensorflow_train_fp16.json  --data_dir $1 --results_dir $2