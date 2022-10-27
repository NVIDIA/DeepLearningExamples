#!/usr/bin/env bash

export OMP_NUM_THREADS=1

# Enables faster cuDNN kernels (available since the 21.12-py3 NGC container)
export CUDNN_V8_API_ENABLED=1  # Keep the flag for older containers
export TORCH_CUDNN_V8_API_ENABLED=1

: ${NUM_GPUS:=8}
: ${BATCH_SIZE:=16}
: ${AMP:=false}
: ${EPOCHS:=6500}
: ${OUTPUT_DIR:="results/hifigan_lj22khz"}
: ${LOG_FILE:=$OUTPUT_DIR/nvlog.json}
: ${DATASET_DIR:="data/LJSpeech-1.1"}
: ${TRAIN_FILELIST:="data/filelists/ljs_audio_train.txt"}
: ${VAL_FILELIST:="data/filelists/ljs_audio_val.txt"}
# Intervals are specified in # of epochs
: ${VAL_INTERVAL:=10}
: ${SAMPLES_INTERVAL:=100}
: ${CHECKPOINT_INTERVAL:=10}
: ${LEARNING_RATE:=0.0003}
: ${LEARNING_RATE_DECAY:=0.9998}
: ${GRAD_ACCUMULATION:=1}
: ${RESUME:=true}

: ${FINE_TUNE_DIR:=""}

mkdir -p "$OUTPUT_DIR"

# Adjust env variables to maintain the global batch size:
#     NUM_GPUS x BATCH_SIZE x GRAD_ACCUMULATION = 128
GBS=$(($NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUMULATION))
[ $GBS -ne 128 ] && echo -e "\nWARNING: Global batch size changed from 128 to ${GBS}."
echo -e "\nAMP=$AMP, ${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}" \
        "(global batch size ${GBS})\n"

ARGS+=" --cuda"
ARGS+=" --dataset_path $DATASET_DIR"
ARGS+=" --training_files $TRAIN_FILELIST"
ARGS+=" --validation_files $VAL_FILELIST"
ARGS+=" --output $OUTPUT_DIR"
ARGS+=" --checkpoint_interval $CHECKPOINT_INTERVAL"
ARGS+=" --epochs $EPOCHS"
ARGS+=" --batch_size $BATCH_SIZE"
ARGS+=" --learning_rate $LEARNING_RATE"
ARGS+=" --lr_decay $LEARNING_RATE_DECAY"
ARGS+=" --validation_interval $VAL_INTERVAL"
ARGS+=" --samples_interval $SAMPLES_INTERVAL"

[ "$AMP" = true ]             && ARGS+=" --amp"
[ "$FINE_TUNE_DIR" != "" ]    && ARGS+=" --input_mels_dir $FINE_TUNE_DIR"
[ "$FINE_TUNE_DIR" != "" ]    && ARGS+=" --fine_tuning"
[ -n "$FINE_TUNE_LR_FACTOR" ] && ARGS+=" --fine_tune_lr_factor $FINE_TUNE_LR_FACTOR"
[ -n "$EPOCHS_THIS_JOB" ]     && ARGS+=" --epochs_this_job $EPOCHS_THIS_JOB"
[ -n "$SEED" ]                && ARGS+=" --seed $SEED"
[ -n "$GRAD_ACCUMULATION" ]   && ARGS+=" --grad_accumulation $GRAD_ACCUMULATION"
[ "$RESUME" = true ]          && ARGS+=" --resume"
[ -n "$LOG_FILE" ]            && ARGS+=" --log_file $LOG_FILE"
[ -n "$BMARK_EPOCHS_NUM" ]    && ARGS+=" --benchmark_epochs_num $BMARK_EPOCHS_NUM"

: ${DISTRIBUTED:="-m torch.distributed.launch --nproc_per_node $NUM_GPUS"}
python $DISTRIBUTED train.py $ARGS "$@"
