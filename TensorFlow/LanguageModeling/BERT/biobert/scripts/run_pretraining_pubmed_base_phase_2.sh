#! /bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${1}
train_batch_size=${2:-16}
learning_rate=${3:-"2.9e-4"}
cased=${4:-false}
precision=${5:-"fp16"}
use_xla=${6:-true}
num_gpu=${7:-16}
warmup_steps=${8:-"434"}
train_steps=${9:-4340}
num_accumulation_steps=${10:-128}
save_checkpoint_steps=${11:-5000}
eval_batch_size=${12:-26}


use_fp16=""
if [ "$precision" = "fp16" ] ; then
        echo "fp16 activated!"
        use_fp16="--use_fp16"
fi

if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--use_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi

if [ "$cased" = "true" ] ; then
    DO_LOWER_CASE=0
    CASING_DIR_PREFIX="cased"
else
    DO_LOWER_CASE=1
    CASING_DIR_PREFIX="uncased"
fi

BERT_CONFIG=/workspace/bert/data/download/google_pretrained_weights/${CASING_DIR_PREFIX}_L-12_H-768_A-12/bert_config.json
RESULTS_DIR=/results
CHECKPOINTS_DIR=${RESULTS_DIR}/biobert_phase_2
mkdir -p ${CHECKPOINTS_DIR}

INPUT_FILES_DIR="/workspace/bert/data/tfrecord/lower_case_${DO_LOWER_CASE}_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/pubmed_baseline/training"
EVAL_FILES_DIR="/workspace/bert/data/tfrecord/lower_case_${DO_LOWER_CASE}_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/pubmed_baseline/test"

if [ $num_gpu -gt 1 ] ; then
    mpi_command="mpirun -np $num_gpu -H localhost:$num_gpu \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib"
    use_hvd="--horovod"
else
    mpi_command=""
    use_hvd=""
fi

export GBS=$(expr $train_batch_size \* $num_gpus \* num_accumulation_steps)
printf -v TAG "tf_bert_bio_1n_phase2_cased_%s_%s_gbs%d" "$cased" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"


$mpi python3 /workspace/bert/run_pretraining.py \
 --input_files_dir=$INPUT_FILES_DIR \
 --eval_files_dir=$EVAL_FILES_DIR \
 --output_dir=$CHECKPOINTS_DIR \
 --bert_config_file=$BERT_CONFIG \
 --do_train=True \
 --do_eval=True \
 --train_batch_size=$train_batch_size \
 --eval_batch_size=$eval_batch_size \
 --max_seq_length=512 \
 --max_predictions_per_seq=80 \
 --num_train_steps=$train_steps \
 --num_warmup_steps=$warmup_steps \
 --save_checkpoints_steps=$save_checkpoint_steps \
 --num_accumulation_steps=$num_accumulation_steps \
 --learning_rate=$learning_rate \
 --report_loss \
 $use_hvd $use_xla_tag $use_fp16 \
 --init_checkpoint=$INIT_CHECKPOINT |& tee $LOGFILE