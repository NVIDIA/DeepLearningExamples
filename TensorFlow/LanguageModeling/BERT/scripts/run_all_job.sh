#!/bin/bash

set -x

# bs64, gpu1, fp32
# bash scripts/run_benchmark.sh 64 1 fp32 50

# bs128, gpu1, fp16
bash scripts/run_benchmark.sh 128 1 fp16 50

# bs32, gpu1, fp32
bash scripts/run_benchmark.sh 32 1 fp32 50

# bs64, gpu1, fp16
bash scripts/run_benchmark.sh 64 1 fp16 50