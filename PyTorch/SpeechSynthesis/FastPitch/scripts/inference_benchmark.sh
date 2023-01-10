#!/usr/bin/env bash

set -a

: ${FILELIST:="phrases/benchmark_8_128.tsv"}
: ${OUTPUT_DIR:="./output/audio_$(basename ${FILELIST} .tsv)"}
: ${TORCHSCRIPT:=true}
: ${BS_SEQUENCE:="1 4 8"}
: ${WARMUP:=64}
: ${REPEATS:=500}
: ${AMP:=false}
: ${CUDNN_BENCHMARK:=true}

for BATCH_SIZE in $BS_SEQUENCE ; do
    LOG_FILE="$OUTPUT_DIR"/perf-infer_amp-${AMP}_bs${BATCH_SIZE}.json
    bash scripts/inference_example.sh "$@"
done
