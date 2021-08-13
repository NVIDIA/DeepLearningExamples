#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="output/FastPitch_checkpoint_1000.pt"}
: ${BATCH_SIZE:=32}
: ${PHRASES:="phrases/devset10.tsv"}
: ${OUTPUT_DIR:="./output/audio_$(basename ${PHRASES} .tsv)"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}

: ${SPEAKER:=0}
: ${NUM_SPEAKERS:=1}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" --cuda"
ARGS+=" --cudnn-benchmark"
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --waveglow $WAVEGLOW"
ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --text-cleaners english_cleaners_v2"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --n-speakers $NUM_SPEAKERS"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"

mkdir -p "$OUTPUT_DIR"

python inference.py $ARGS "$@"
