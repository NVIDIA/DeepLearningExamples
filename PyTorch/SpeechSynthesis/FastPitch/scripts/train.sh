#!/bin/bash

# Default recipe for 8x GPU 16GB with TensorCores (fp16/AMP).
# For other configurations, adjust
#
#     batch-size x graient-accumulation-steps
#
# to maintain a total of 64x4=256 samples per step.
#
#   | Prec. | #GPU | -bs | --gradient-accumulation-steps |
#   |-------|------|-----|-------------------------------|
#   | AMP   |    1 |  64 |                             4 |
#   | AMP   |    4 |  64 |                             1 |
#   | AMP   |    8 |  32 |                             1 |
#   | FP32  |    1 |  32 |                             8 |
#   | FP32  |    4 |  32 |                             2 |
#   | FP32  |    8 |  32 |                             1 |

mkdir -p output
python -m multiproc train.py \
    --cuda \
    --cudnn-enabled \
    -o ./output/ \
    --log-file ./output/nvlog.json \
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
    --gradient-accumulation-steps 1 \
    --amp-run
