#!/usr/bin/env bash

DATA_DIR="LJSpeech-1.1"
EXP_DIR="output"
WAVEG_CH="pretrained_models/waveglow/waveglow_256channels_ljs_v3.pt"

CHECKPOINT=${1:-1500}

python inference.py -i phrases/devset10.tsv \
                    -o ${EXP_DIR}/audio_devset10_checkpoint${CHECKPOINT} \
                    --log-file ${EXP_DIR}/nvlog_inference.json \
                    --dataset-path ${DATA_DIR} \
                    --fastpitch ${EXP_DIR}/checkpoint_FastPitch_${CHECKPOINT}.pt \
                    --waveglow ${WAVEG_CH} \
		    --wn-channels 256 \
                    --batch-size 32 \
                    --amp-run \
                    --cuda
