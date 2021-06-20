#! /bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

echo "Container nvidia build = " $NVIDIA_BUILD_ID

num_gpus=${1:-8}
train_batch_size=${2:-14}
learning_rate=${3:-"1e-4"}
precision=${4:-"fp16"}
use_xla=${5:-"true"}
warmup_steps=${6:-"10000"}
train_steps=${7:-1144000}
bert_model=${8:-"large"}
num_accumulation_steps=${9:-1}
seq_len=${10:-512}
max_pred_per_seq=${11:-80}

DATA_DIR=data/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus

if [ "$bert_model" = "large" ] ; then
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json
else
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--use_fp16"
elif [ "$precision" = "fp32" ] || [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$use_xla" = "true" ] ; then
    PREC="$PREC --enable_xla"
    echo "XLA activated"
fi

export GBS=$(expr $train_batch_size \* $num_gpus)
printf -v TAG "tf_bert_pretraining_adam_%s_%s_gbs%d" "$bert_model" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results/${TAG}_${DATESTAMP}}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

INPUT_FILES="$DATA_DIR/training/*"
EVAL_FILES="$DATA_DIR/test"

CMD="python3 run_pretraining.py"
CMD+=" --input_files=$INPUT_FILES"
CMD+=" --model_dir=$RESULTS_DIR"
CMD+=" --bert_config_file=$BERT_CONFIG"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=$seq_len"
CMD+=" --max_predictions_per_seq=$max_pred_per_seq"
CMD+=" --num_steps_per_epoch=$train_steps --num_train_epochs=1"
CMD+=" --warmup_steps=$warmup_steps"
CMD+=" --num_accumulation_steps=$num_accumulation_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" $PREC"

#Check if all necessary files are available before training
for DIR_or_file in $DATA_DIR $BERT_CONFIG $RESULTS_DIR; do
  if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
     echo "Error! $DIR_or_file directory missing. Please mount correctly"
     exit -1
  fi
done

if [ $num_gpus -gt 1 ] ; then
   mpi="mpirun -np $num_gpus \
   --allow-run-as-root -bind-to none -map-by slot \
   -x NCCL_DEBUG=INFO \
   -x LD_LIBRARY_PATH \
   -x PATH -mca pml ob1 -mca btl ^openib"
   CMD="$mpi $CMD --use_horovod"
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

