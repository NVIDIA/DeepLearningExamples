#!/bin/bash

[ ! -n "$WAVEG_CH" ] && WAVEG_CH="pretrained_models/waveglow/waveglow_1076430_14000_amp.pt"
[ ! -n "$FASTPITCH_CH" ] && FASTPITCH_CH="output/FastPitch_checkpoint_1500.pt"
[ ! -n "$REPEATS" ] && REPEATS=1000
[ ! -n "$BS_SEQ" ] && BS_SEQ="1 4 8"
[ ! -n "$PHRASES" ] && PHRASES="phrases/benchmark_8_128.tsv"
[ ! -n "$OUTPUT_DIR" ] && OUTPUT_DIR="./output/audio_$(basename ${PHRASES} .tsv)"
[ "$AMP" == "true" ] && AMP_FLAG="--amp" || AMP=false
[ "$SET_AFFINITY" == "true" ] && SET_AFFINITY_FLAG="--set-affinity"

for BS in $BS_SEQ ; do

  echo -e "\nAMP: ${AMP}, batch size: ${BS}\n"

  python inference.py --cuda \
                      -i ${PHRASES} \
                      -o ${OUTPUT_DIR} \
                      --fastpitch ${FASTPITCH_CH} \
                      --waveglow ${WAVEG_CH} \
                      --wn-channels 256 \
                      --include-warmup \
                      --batch-size ${BS} \
                      --repeats ${REPEATS} \
                      --torchscript \
                      ${AMP_FLAG} ${SET_AFFINITY_FLAG}
done
