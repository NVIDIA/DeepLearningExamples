#!/bin/bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="output/FastPitch_checkpoint_1500.pt"}
: ${REPEATS:=1000}
: ${BS_SEQUENCE:="1 4 8"}
: ${PHRASES:="phrases/benchmark_8_128.tsv"}
: ${OUTPUT_DIR:="./output/audio_$(basename ${PHRASES} .tsv)"}
: ${AMP:=false}

[ "$AMP" = true ] && AMP_FLAG="--amp"

mkdir -o "$OUTPUT_DIR"

for BS in $BS_SEQUENCE ; do

  echo -e "\nAMP: ${AMP}, batch size: ${BS}\n"

  python inference.py --cuda --cudnn-benchmark \
                      -i ${PHRASES} \
                      -o ${OUTPUT_DIR} \
                      --fastpitch ${FASTPITCH} \
                      --waveglow ${WAVEGLOW} \
                      --wn-channels 256 \
                      --include-warmup \
                      --batch-size ${BS} \
                      --repeats ${REPEATS} \
                      --torchscript \
                      ${AMP_FLAG}
done
