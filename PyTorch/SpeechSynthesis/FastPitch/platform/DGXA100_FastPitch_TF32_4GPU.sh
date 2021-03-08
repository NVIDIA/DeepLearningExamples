#!/bin/bash

mkdir -p output
python -m torch.distributed.launch --nproc_per_node 4 train.py \
    --cuda \
    -o ./output/ \
    --log-file output/nvlog.json \
    --dataset-path LJSpeech-1.1 \
    --training-files filelists/ljs_mel_dur_pitch_text_train_filelist.txt \
    --validation-files filelists/ljs_mel_dur_pitch_text_test_filelist.txt \
    --pitch-mean-std-file LJSpeech-1.1/pitch_char_stats__ljs_audio_text_train_filelist.json \
    --epochs 1500 \
    --epochs-per-checkpoint 100 \
    --warmup-steps 1000 \
    -lr 0.1 \
    -bs 32 \
    --optimizer lamb \
    --grad-clip-thresh 1000.0 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --weight-decay 1e-6 \
    --gradient-accumulation-steps 2
