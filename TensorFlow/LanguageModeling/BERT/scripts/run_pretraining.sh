#! /bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

WIKI_DIR=/workspace/bert/data/wikipedia_corpus/final_tfrecords_sharded
BOOKS_DIR=/workspace/bert/data/bookcorpus/final_tfrecords_sharded
BERT_CONFIG=/workspace/bert/data/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_config.json

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results

if [ ! -d "$WIKI_DIR" ] ; then
   echo "Error! $WIKI_DIR directory missing. Please mount wikipedia dataset."
   exit -1
else
   SOURCES="$WIKI_DIR/*"
fi
if [ ! -d "$BOOKS_DIR" ] ; then
   echo "Warning! $BOOKS_DIR directory missing. Training will proceed without book corpus."
else
   SOURCES+=" $BOOKS_DIR/*"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi

if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

train_batch_size=${1:-14}
eval_batch_size=${2:-8}
learning_rate=${3:-"1e-4"}
precision=${4:-"manual_fp16"}
use_xla=${5:-"true"}
num_gpus=${6:-1}
warmup_steps=${7:-"10000"}
train_steps=${8:-1144000}
save_checkpoints_steps=${9:-5000}

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--use_fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "manual_fp16" ] ; then
   PREC="--manual_fp16"
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$use_xla" = "true" ] ; then
    PREC="$PREC --use_xla"
    echo "XLA activated"
fi

export GBS=$(expr $train_batch_size \* $num_gpus)
printf -v TAG "tf_bert_pretraining_%s_gbs%d" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`
RESULTS_DIR=${RESULTS_DIR}/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

echo $SOURCES
INPUT_FILES=$(eval ls $SOURCES | tr " " "\n" | awk '{printf "%s,",$1}' | sed s'/.$//')
CMD="python3 /workspace/bert/run_pretraining.py"
CMD+=" --input_file=$INPUT_FILES"
CMD+=" --output_dir=$RESULTS_DIR"
CMD+=" --bert_config_file=$BERT_CONFIG"
CMD+=" --do_train=True"
CMD+=" --do_eval=True"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --eval_batch_size=$eval_batch_size"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --num_train_steps=$train_steps"
CMD+=" --num_warmup_steps=$warmup_steps"
CMD+=" --save_checkpoints_steps=$save_checkpoints_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --report_loss"
CMD+=" --horovod $PREC"

if [ $num_gpus -gt 1 ] ; then
   CMD="mpiexec --allow-run-as-root -np $num_gpus --bind-to socket $CMD"
fi




set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi
set +x
