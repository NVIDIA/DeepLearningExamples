#!/bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID
train_batch_size=${1:-14}
learning_rate=${2:-"0.4375e-4"}
precision=${3:-"fp16"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.01"}
train_steps=${6:-2285714}
save_checkpoint_steps=${7:-2000}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"false"}
gradient_accumulation_steps=${11:-1}
seed=${12:-42}
job_name=${13:-"job"}


DATASET=wikipedia_corpus # change this for other datasets

DATA_DIR=data/${DATASET}/hdf5_shards/
BERT_CONFIG=bert_config.json
RESULTS_DIR=/results
CHECKPOINTS_DIR=/results/checkpoints

mkdir -p $CHECKPOINTS_DIR


if [ ! -d "$DATA_DIR" ] ; then
   echo "Warning! $DATA_DIR directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

echo $DATA_DIR
INPUT_DIR=$DATA_DIR
CMD=" /workspace/bert/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"


if [ "$num_gpus" -gt 1  ] ; then
   CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"
else
   CMD="python3  $CMD"
fi


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
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

echo "finished pretraining, starting benchmarking"

target_loss=15
THROUGHPUT=10
THRESHOLD=0.9

throughput=`cat $LOGFILE | grep Iteration | tail -1 | awk -F's/it' '{print $1}' | awk -F',' '{print $2}' | egrep -o [0-9.]+`
loss=`cat $LOGFILE | grep 'Average Loss' | tail -1 | awk -F'Average Loss =' '{print $2}' | awk -F' ' '{print $1}' | egrep -o [0-9.]+`
final_loss=`cat $LOGFILE | grep 'Total Steps' | tail -1 | awk -F'Final Loss =' '{print $2}' | awk -F' ' '{print $1}' | egrep -o [0-9.]+`

echo "throughput: $throughput s/it"
echo "average loss: $loss"
echo "final loss: $final_loss"

ACCURACY_TEST_RESULT=$(awk 'BEGIN {print ('${loss}' <= '${target_loss}')}')

if [ $ACCURACY_TEST_RESULT == 1 ];
    then
        echo "&&&& ACCURACY TEST PASSED"
    else
        echo "&&&& ACCURACY TEST FAILED"
    fi

PERFORMANCE_TEST_RESULT=$(awk 'BEGIN {print ('${throughput}' <= ('${THROUGHPUT}' * '${THRESHOLD}'))}')

if [ $PERFORMANCE_TEST_RESULT == 1 ];
    then
        echo "&&&& PERFORMANCE TEST PASSED"
    else
        echo "&&&& PERFORMANCE TEST FAILED"
    fi

if [ $ACCURACY_TEST_RESULT == 1 -a $PERFORMANCE_TEST_RESULT == 1 ];
    then
        echo "&&&& PASSED"
        exit 0
    else
        echo "&&&& FAILED"
        exit 1
    fi


