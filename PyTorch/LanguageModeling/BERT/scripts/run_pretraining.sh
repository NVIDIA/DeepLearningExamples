#!/bin/bash

# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
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

echo "Container nvidia build = " $NVIDIA_BUILD_ID
train_batch_size=${1:-8192}
learning_rate=${2:-"6e-3"}
precision=${3:-"fp16"}
num_gpus=${4:-$(nvidia-smi -L | wc -l)}
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-7038}
save_checkpoint_steps=${7:-200}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-128}
seed=${12:-12439}
job_name=${13:-"bert_lamb_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
train_batch_size_phase2=${16:-4096}
learning_rate_phase2=${17:-"4e-3"}
warmup_proportion_phase2=${18:-"0.128"}
train_steps_phase2=${19:-1563}
gradient_accumulation_steps_phase2=${20:-512}
#change this for other datasets
DATASET=pretrain/phase1/unbinned/parquet
DATA_DIR_PHASE1=${21:-$BERT_PREP_WORKING_DIR/${DATASET}/}
#change this for other datasets
DATASET2=pretrain/phase2/bin_size_64/parquet
DATA_DIR_PHASE2=${22:-$BERT_PREP_WORKING_DIR/${DATASET2}/}
CODEDIR=${23:-"/workspace/bert"}
init_checkpoint=${24:-"None"}
VOCAB_FILE=vocab/vocab
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints
wikipedia_source=${25:-$BERT_PREP_WORKING_DIR/wikipedia/source/}
num_dask_workers=${26:-$(nproc)}
num_shards_per_worker=${27:-128}
num_workers=${28:-4}
num_nodes=1
sample_ratio=${29:-0.9}
phase2_bin_size=${30:-64}
masking=${31:-static}
BERT_CONFIG=${32:-bert_configs/large.json}

# Calculate the total number of shards.
readonly num_blocks=$((num_shards_per_worker * $(( num_workers > 0 ? num_workers : 1 )) * num_nodes * num_gpus))

if [ "${phase2_bin_size}" == "none" ]; then
   readonly phase2_bin_size_flag=""
elif [[ "${phase2_bin_size}" =~ ^(32|64|128|256|512)$ ]]; then
   readonly phase2_bin_size_flag="--bin-size ${phase2_bin_size}"
else
   echo "Error! phase2_bin_size=${phase2_bin_size} not supported!"
   return -1
fi

if [ "${masking}" == "static" ]; then
   readonly masking_flag="--masking"
elif [ "${masking}" == "dynamic" ]; then
   readonly masking_flag=""
else
   echo "Error! masking=${masking} not supported!"
   return -1
fi

mkdir -p $CHECKPOINTS_DIR


if [ ! -d "${DATA_DIR_PHASE1}" ] || [ -z "$(ls -A ${DATA_DIR_PHASE1})" ]; then
   echo "Warning! ${DATA_DIR_PHASE1} directory missing."
   if [ ! -d "${wikipedia_source}" ] || [ -z "$(ls -A ${wikipedia_source})" ]; then
      echo "Error! ${wikipedia_source} directory missing. Training cannot start!"
      return -1
   fi
   preprocess_cmd=" \
      mpirun \
         --oversubscribe \
         --allow-run-as-root \
         -np ${num_dask_workers} \
         -x LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so \
            preprocess_bert_pretrain \
               --schedule mpi \
               --vocab-file ${VOCAB_FILE} \
               --wikipedia ${wikipedia_source} \
               --sink ${DATA_DIR_PHASE1} \
               --num-blocks ${num_blocks} \
               --sample-ratio ${sample_ratio} \
               ${masking_flag} \
               --seed ${seed}"
   echo "Running ${preprocess_cmd} ..."
   ${preprocess_cmd}

   balance_load_cmd=" \
      mpirun \
         --oversubscribe \
         --allow-run-as-root \
         -np ${num_dask_workers} \
            balance_dask_output \
               --indir ${DATA_DIR_PHASE1} \
               --num-shards ${num_blocks}"
   echo "Running ${balance_load_cmd} ..."
   ${balance_load_cmd}
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
elif [ "$precision" = "tf32" ] ; then
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

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --vocab_file=$VOCAB_FILE"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "
CMD+=" --disable_progress_bar"
CMD+=" --num_workers=${num_workers}"

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
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

echo "finished pretraining"

#Start Phase2

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

if [ ! -d "${DATA_DIR_PHASE2}" ] || [ -z "$(ls -A ${DATA_DIR_PHASE2})" ]; then
   echo "Warning! ${DATA_DIR_PHASE2} directory missing."
   if [ ! -d "${wikipedia_source}" ] || [ -z "$(ls -A ${wikipedia_source})" ]; then
      echo "Error! ${wikipedia_source} directory missing. Training cannot start!"
      return -1
   fi
   preprocess_cmd=" \
      mpirun \
         --oversubscribe \
         --allow-run-as-root \
         -np ${num_dask_workers} \
         -x LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so \
            preprocess_bert_pretrain \
               --schedule mpi \
               --vocab-file ${VOCAB_FILE} \
               --wikipedia ${wikipedia_source} \
               --sink ${DATA_DIR_PHASE2} \
               --target-seq-length 512 \
               --num-blocks ${num_blocks} \
               --sample-ratio ${sample_ratio} \
               ${phase2_bin_size_flag} \
               ${masking_flag} \
               --seed ${seed}"
   echo "Running ${preprocess_cmd} ..."
   ${preprocess_cmd}

   balance_load_cmd=" \
      mpirun \
         --oversubscribe \
         --allow-run-as-root \
         -np ${num_dask_workers} \
            balance_dask_output \
               --indir ${DATA_DIR_PHASE2} \
               --num-shards ${num_blocks}"
   echo "Running ${balance_load_cmd} ..."
   ${balance_load_cmd}
fi
echo $DATA_DIR_PHASE2
INPUT_DIR=$DATA_DIR_PHASE2
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --vocab_file=$VOCAB_FILE"
CMD+=" --train_batch_size=$train_batch_size_phase2"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" --do_train --phase2 --resume_from_checkpoint --phase1_end_step=$train_steps"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "
CMD+=" --disable_progress_bar"
CMD+=" --num_workers=${num_workers}"

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
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

echo "finished phase2"

