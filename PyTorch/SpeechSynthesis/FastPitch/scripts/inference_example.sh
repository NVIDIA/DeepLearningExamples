#!/usr/bin/env bash

export CUDNN_V8_API_ENABLED=1

: ${DATASET_DIR:="data/LJSpeech-1.1"}
: ${BATCH_SIZE:=32}
: ${FILELIST:="phrases/devset10.tsv"}
: ${AMP:=false}
: ${TORCHSCRIPT:=true}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

# Mel-spectrogram generator (optional)
: ${FASTPITCH="pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"}

# Vocoder; set only one
: ${WAVEGLOW="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${HIFIGAN=""}

# Synthesis
: ${SPEAKER:=0}
: ${DENOISING:=0.01}

if [ ! -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./output/audio_$(basename ${FILELIST} .tsv)"
    [ "$AMP" = true ]     && OUTPUT_DIR+="_fp16"
    [ "$AMP" = false ]    && OUTPUT_DIR+="_fp32"
    [ -n "$FASTPITCH" ]   && OUTPUT_DIR+="_fastpitch"
    [ ! -n "$FASTPITCH" ] && OUTPUT_DIR+="_gt-mel"
    [ -n "$WAVEGLOW" ]    && OUTPUT_DIR+="_waveglow"
    [ -n "$HIFIGAN" ]     && OUTPUT_DIR+="_hifigan"
    OUTPUT_DIR+="_denoise-"${DENOISING}
fi
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
mkdir -p "$OUTPUT_DIR"

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" --cuda"
ARGS+=" --cudnn-benchmark"
ARGS+=" --dataset-path $DATASET_DIR"
ARGS+=" -i $FILELIST"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --repeats $REPEATS"
ARGS+=" --speaker $SPEAKER"
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]         && ARGS+=" --amp"
[ "$TORCHSCRIPT" = true ] && ARGS+=" --torchscript"
[ -n "$HIFIGAN" ]         && ARGS+=" --hifigan $HIFIGAN"
[ -n "$WAVEGLOW" ]        && ARGS+=" --waveglow $WAVEGLOW"
[ -n "$FASTPITCH" ]       && ARGS+=" --fastpitch $FASTPITCH"

python inference.py $ARGS "$@"
