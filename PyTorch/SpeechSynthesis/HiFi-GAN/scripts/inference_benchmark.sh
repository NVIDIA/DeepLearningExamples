#!/usr/bin/env bash

export CUDNN_V8_API_ENABLED=1  # Keep the flag for older containers
export TORCH_CUDNN_V8_API_ENABLED=1

set -a

: ${AMP:=false}
: ${CUDNN_BENCHMARK:=true}
: ${FILELIST:="data/filelists/benchmark_8_128.tsv"}
: ${OUTPUT_DIR:="./results"}
: ${TORCHSCRIPT:=true}
: ${REPEATS:=200}
: ${WARMUP:=100}
: ${DENOISING:=0.0}
: ${BATCH_SIZE:=1}  # 1 2 4 8

LOG_FILE="$OUTPUT_DIR"/perf-infer_amp-${AMP}_bs${BATCH_SIZE}
LOG_FILE+=_denoising${DENOISING}
LOG_FILE+=_torchscript-${TORCHSCRIPT}
LOG_FILE+=.json
bash scripts/inference_example.sh "$@"
