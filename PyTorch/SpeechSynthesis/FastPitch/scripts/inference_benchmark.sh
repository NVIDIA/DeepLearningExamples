#!/bin/bash

MODEL_DIR="pretrained_models"
EXP_DIR="output"

WAVEG_CH="waveglow_256channels_ljs_v3.pt"

BSZ=${1:-4}
PRECISION=${2:-fp16}

for PRECISION in fp16 fp32; do
  for BSZ in 1 4 8 ; do

    echo -e "\nprecision=${PRECISION} batch size=${BSZ}\n"

    [ "$PRECISION" == "fp16" ] && AMP_FLAG="--amp-run" || AMP_FLAG=""

    python inference.py --cuda --wn-channels 256 ${AMP_FLAG} \
                        --fastpitch ${EXP_DIR}/checkpoint_FastPitch_1500.pt \
                        --waveglow ${MODEL_DIR}/waveglow/${WAVEG_CH} \
                        --include-warmup \
                        --batch-size ${BSZ} \
                        --repeats 1000 \
                        -i phrases/benchmark_8_128.tsv
  done
done
