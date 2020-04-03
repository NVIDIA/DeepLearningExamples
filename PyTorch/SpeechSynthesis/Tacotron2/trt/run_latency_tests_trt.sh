#!/bin/bash

for i in {1..1003}
do
    python trt/inference_trt.py -i ./phrases/phrase_1_128.txt --encoder ./output/encoder_fp16.engine --decoder ./output/decoder_iter_fp16.engine --postnet ./output/postnet_fp16.engine  --waveglow ./output/waveglow_fp16.engine -o output/ --fp16 >> tmp_log_bs1_fp16.log 2>&1
done
