#!/usr/bin/env bash

set -a

: ${PHRASES:="phrases/benchmark_8_128.tsv"}
: ${OUTPUT_DIR:="./output/audio_$(basename ${PHRASES} .tsv)"}
: ${TORCHSCRIPT:=true}
: ${REPEATS:=100}
: ${BS_SEQUENCE:="1 4 8"}
: ${WARMUP:=100}

for BATCH_SIZE in $BS_SEQUENCE ; do
    LOG_FILE="$OUTPUT_DIR"/perf-infer_amp-${AMP}_bs${BATCH_SIZE}.json
    bash scripts/inference_example.sh "$@"
done
