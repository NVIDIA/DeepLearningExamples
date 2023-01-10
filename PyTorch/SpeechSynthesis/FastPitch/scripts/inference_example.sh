#!/usr/bin/env bash

export CUDNN_V8_API_ENABLED=1  # Keep the flag for older containers
export TORCH_CUDNN_V8_API_ENABLED=1

: ${DATASET_DIR:="data/LJSpeech-1.1"}
: ${BATCH_SIZE:=32}
: ${FILELIST:="phrases/devset10.tsv"}
: ${AMP:=false}
: ${TORCHSCRIPT:=true}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}
: ${PHONE:=true}
: ${CUDNN_BENCHMARK:=false}

# Paths to pre-trained models downloadable from NVIDIA NGC (LJSpeech-1.1)
FASTPITCH_LJ="pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"
HIFIGAN_LJ="pretrained_models/hifigan/hifigan_gen_checkpoint_10000_ft.pt"
WAVEGLOW_LJ="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"

# Mel-spectrogram generator (optional; can synthesize from ground-truth spectrograms)
: ${FASTPITCH=$FASTPITCH_LJ}

# Vocoder (set only one)
: ${HIFIGAN=$HIFIGAN_LJ}
# : ${WAVEGLOW=$WAVEGLOW_LJ}

[[ "$FASTPITCH" == "$FASTPITCH_LJ" && ! -f "$FASTPITCH" ]] && { echo "Downloading $FASTPITCH from NGC..."; bash scripts/download_models.sh fastpitch; }
[[ "$WAVEGLOW" == "$WAVEGLOW_LJ" && ! -f "$WAVEGLOW" ]] && { echo "Downloading $WAVEGLOW from NGC..."; bash scripts/download_models.sh waveglow; }
[[ "$HIFIGAN" == "$HIFIGAN_LJ" && ! -f "$HIFIGAN" ]] && { echo "Downloading $HIFIGAN from NGC..."; bash scripts/download_models.sh hifigan-finetuned-fastpitch; }

if [[ "$HIFIGAN" == "$HIFIGAN_LJ" && "$FASTPITCH" != "$FASTPITCH_LJ" ]]; then
    echo -e "\nNOTE: Using HiFi-GAN checkpoint trained for the LJSpeech-1.1 dataset."
    echo -e "NOTE: If you're using a different dataset, consider training a new HiFi-GAN model or switch to WaveGlow."
    echo -e "NOTE: See $0 for details.\n"
fi

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

ARGS+=" --cuda"
ARGS+=" --dataset-path $DATASET_DIR"
ARGS+=" -i $FILELIST"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --repeats $REPEATS"
ARGS+=" --speaker $SPEAKER"
[ "$CPU" = false ]        && ARGS+=" --cuda"
[ "$AMP" = true ]         && ARGS+=" --amp"
[ "$TORCHSCRIPT" = true ] && ARGS+=" --torchscript"
[ -n "$HIFIGAN" ]         && ARGS+=" --hifigan $HIFIGAN"
[ -n "$WAVEGLOW" ]        && ARGS+=" --waveglow $WAVEGLOW"
[ -n "$FASTPITCH" ]       && ARGS+=" --fastpitch $FASTPITCH"
[ "$PHONE" = true ]       && ARGS+=" --p-arpabet 1.0"
[[ "$CUDNN_BENCHMARK" = true && "$CPU" = false ]] && ARGS+=" --cudnn-benchmark"

python inference.py $ARGS "$@"
