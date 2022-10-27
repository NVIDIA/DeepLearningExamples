#!/usr/bin/env bash

# Runs a process which resembles FastPitch training and extracts mel-scale
# spectrograms generated with FastPitch for HiFi-GAN fine-tuning.

export OMP_NUM_THREADS=1

: ${NUM_GPUS:=8}
: ${BATCH_SIZE:=16}
: ${AMP:=false}

: ${DATASET_PATH:="data/LJSpeech-1.1"}
: ${OUTPUT_DIR:="data/mels-fastpitch-ljs22khz"}
: ${DATASET_FILELIST:=data/filelists/ljs_audio_pitch_text.txt}  # train + val + test
: ${LOAD_PITCH_FROM_DISK:=true}
: ${LOAD_MEL_FROM_DISK:=false}  # mel-spec of the original data
: ${SAMPLING_RATE:=22050}
: ${FASTPITCH:="pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"}

mkdir -p "$OUTPUT_DIR"

# Pre-calculate pitch values and write to disk
# This step requires only CPU
if [[ "$LOAD_PITCH_FROM_DISK" = true && ! -d "$DATASET_PATH/pitch" ]]; then

  echo "Pitch values needs for FastPitch not found in $DATASET_PATH/pitch."
  echo "Calcluating..."

  python prepare_dataset.py \
      --wav-text-filelists data/filelists/ljs_audio_text.txt \
      --n-workers 16 \
      --batch-size 1 \
      --dataset-path $DATASET_PATH \
      --extract-pitch
fi

ARGS+=" --cuda"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --dataset-path $DATASET_PATH"
ARGS+=" --dataset-files $DATASET_FILELIST"
ARGS+=" -bs $BATCH_SIZE"

[ -n "$FASTPITCH" ]                && ARGS+=" --checkpoint-path $FASTPITCH"
[ -z "$FASTPITCH" ]                && ARGS+=" --resume"
[ "$AMP" = "true" ]                && ARGS+=" --amp"
[ "$LOAD_MEL_FROM_DISK" = true ]   && ARGS+=" --load-mel-from-disk"
[ "$LOAD_PITCH_FROM_DISK" = true ] && ARGS+=" --load-pitch-from-disk"
[ "$PITCH_ONLINE_DIR" != "" ]      && ARGS+=" --pitch-online-dir $PITCH_ONLINE_DIR"  # e.g., /dev/shm/pitch

if [ "$SAMPLING_RATE" == "44100" ]; then
  ARGS+=" --sampling-rate 44100"
  ARGS+=" --filter-length 2048"
  ARGS+=" --hop-length 512"
  ARGS+=" --win-length 2048"
  ARGS+=" --mel-fmin 0.0"
  ARGS+=" --mel-fmax 22050.0"

elif [ "$SAMPLING_RATE" != "22050" ]; then
  echo "Sampling rate $SAMPLING_RATE not supported. Edit $0 manually."
  exit 1
fi

: ${DISTRIBUTED:="-m torch.distributed.launch --nproc_per_node $NUM_GPUS"}
python $DISTRIBUTED fastpitch/extract_mels.py $ARGS "$@"

