#!/bin/bash

export OMP_NUM_THREADS=1

: ${NUM_GPUS:=8}
: ${BS:=32}
: ${GRAD_ACCUMULATION:=1}
: ${OUTPUT_DIR:="./output"}
: ${AMP:=false}
: ${EPOCHS:=1500}

[ "$AMP" == "true" ] && AMP_FLAG="--amp"

# Adjust env variables to maintain the global batch size
#
#    NGPU x BS x GRAD_ACC = 256.
#
GBS=$(($NUM_GPUS * $BS * $GRAD_ACCUMULATION))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}.\n"

echo -e "\nSetup: ${NUM_GPUS}x${BS}x${GRAD_ACCUMULATION} - global batch size ${GBS}\n"

mkdir -p "$OUTPUT_DIR"
python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} train.py \
    --cuda \
    -o "$OUTPUT_DIR/" \
    --log-file "$OUTPUT_DIR/nvlog.json" \
    --dataset-path LJSpeech-1.1 \
    --training-files filelists/ljs_mel_dur_pitch_text_train_filelist.txt \
    --validation-files filelists/ljs_mel_dur_pitch_text_test_filelist.txt \
    --pitch-mean-std-file LJSpeech-1.1/pitch_char_stats__ljs_audio_text_train_filelist.json \
    --epochs ${EPOCHS} \
    --epochs-per-checkpoint 100 \
    --warmup-steps 1000 \
    -lr 0.1 \
    -bs ${BS} \
    --optimizer lamb \
    --grad-clip-thresh 1000.0 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --weight-decay 1e-6 \
    --gradient-accumulation-steps ${GRAD_ACCUMULATION} \
    ${AMP_FLAG}
