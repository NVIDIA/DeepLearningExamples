#!/bin/bash

set -e

## uncomment to generate new baseline; will be created in qa/baselines/ ##
## python inference_perf.py -m Tacotron2 -bs=20 --input-text qa/text_padded.pt  --create-benchmark

python inference_perf.py -m Tacotron2 -bs=1 --decoder-no-early-stopping --input-text qa/text_padded.pt
python qa/check_curves.py \
    -b qa/tacotron2_fp32-infer-bs1.json \
    Tacotron2_infer_BS1_FP32_DGX1_16GB_1GPU_single.json \
    -g "iter" \
    -k "items_per_sec" \
    --eps 0.001 \
    --damping 1 \
    --sigma 12.0 \
    --fail low
