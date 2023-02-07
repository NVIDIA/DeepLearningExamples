# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

set -ex

echo "Container nvidia build = " $NVIDIA_BUILD_ID
train_batch_size=${1:-256}
learning_rate=${2:-"6e-3"}
precision=${3:-"amp"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-7038}
save_checkpoint_steps=${7:-200}
create_logfile=${8:-"false"}
gradient_accumulation_steps=${9:-32}
seed=${10:-12439}
job_name=${11:-"bert_lamb_pretraining"}
train_batch_size_phase2=${12:-32}
learning_rate_phase2=${13:-"4e-3"}
warmup_proportion_phase2=${14:-"0.128"}
train_steps_phase2=${15:-1563}
gradient_accumulation_steps_phase2=${16:-128}
#change this for other datasets
DATASET=pretrain/phase1/unbinned/parquet
DATA_DIR_PHASE1=${17:-$BERT_PREP_WORKING_DIR/${DATASET}/}
#change this for other datasets
DATASET2=pretrain/phase2/bin_size_64/parquet
DATA_DIR_PHASE2=${18:-$BERT_PREP_WORKING_DIR/${DATASET2}/}
CODEDIR=${19:-"/workspace/bert"}
init_checkpoint=${20:-"None"}
VOCAB_FILE=vocab/bert-large-uncased-vocab.txt
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR
wikipedia_source=${21:-$BERT_PREP_WORKING_DIR/wikipedia/source/}
num_dask_workers=${22:-$(nproc)}
num_shards_per_worker=${23:-128}
num_workers=${24:-4}
num_nodes=1
sample_ratio=${25:-0.9}
phase2_bin_size=${26:-64}
masking=${27:-static}
BERT_CONFIG=${28:-"None"}
enable_benchmark=${29:-"false"}
benchmark_steps=${30:-"10"}
benchmark_warmup_steps=${31:-"10"}

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

CONFIG=""
if [ "$BERT_CONFIG" != "None" ] ; then
  CONFIG="--config-file=$BERT_CONFIG"
fi

PREC=""
if [ "$precision" = "amp" ] ; then
   PREC="--amp --use-dynamic-loss-scaling --scale-loss=1048576"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS="--gradient-merge-steps=$gradient_accumulation_steps"


INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--from-checkpoint=$init_checkpoint --last-step-of-checkpoint=auto"
fi

BENCH=""
if [ "$enable_benchmark" = "true" ] ; then
   BENCH="--benchmark --benchmark-steps=$benchmark_steps --benchmark-warmup-steps=$benchmark_warmup_steps"
fi


unset CUDA_VISIBLE_DEVICES
if [ "$num_gpus" = "1" ] ; then
  DIST_CMD="python -m paddle.distributed.launch --gpus=0"
elif [ "$num_gpus" = "2" ] ; then
  DIST_CMD="python -m paddle.distributed.launch --gpus=0,1"
elif [ "$num_gpus" = "3" ] ; then
  DIST_CMD="python -m paddle.distributed.launch --gpus=0,1,2"
elif [ "$num_gpus" = "4" ] ; then
  DIST_CMD="python -m paddle.distributed.launch --gpus=0,1,2,3"
elif [ "$num_gpus" = "5" ] ; then
  DIST_CMD="python -m paddle.distributed.launch --gpus=0,1,2,3,4"
elif [ "$num_gpus" = "6" ] ; then
  DIST_CMD="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5"
elif [ "$num_gpus" = "7" ] ; then
  DIST_CMD="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6"
elif [ "$num_gpus" = "8" ] ; then
  DIST_CMD="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7"
else
  echo "Wrong number of gpus"
  exit -2
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input-dir=$DATA_DIR_PHASE1"
CMD+=" --vocab-file=$VOCAB_FILE"
CMD+=" --output-dir=$CHECKPOINTS_DIR"
CMD+=" $CONFIG "
CMD+=" --bert-model=bert-large-uncased"
CMD+=" --batch-size=$train_batch_size"
CMD+=" --max-seq-length=128"
CMD+=" --max-predictions-per-seq=20"
CMD+=" --max-steps=$train_steps"
CMD+=" --warmup-proportion=$warmup_proportion"
CMD+=" --num-steps-per-checkpoint=$save_checkpoint_steps"
CMD+=" --learning-rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" --log-freq=1"
CMD+=" --optimizer=Lamb"
CMD+=" --phase1"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $INIT_CHECKPOINT"
CMD+=" $BENCH"
CMD+=" --report-file ${RESULTS_DIR}/dllogger_p1.json "

CMD="$DIST_CMD $CMD"


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus \* $gradient_accumulation_steps)
  printf -v TAG "paddle_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
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
if [ "$precision" = "amp" ] ; then
   PREC="--amp --use-dynamic-loss-scaling --scale-loss=1048576"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS="--gradient-merge-steps=$gradient_accumulation_steps_phase2"


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
PHASE1_END_CKPT_DIR="${CHECKPOINTS_DIR}/bert-large-uncased/phase1/${train_steps}"
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input-dir=$DATA_DIR_PHASE2"
CMD+=" --vocab-file=$VOCAB_FILE"
CMD+=" --output-dir=$CHECKPOINTS_DIR"
CMD+=" $CONFIG "
CMD+=" --bert-model=bert-large-uncased"
CMD+=" --batch-size=$train_batch_size_phase2"
CMD+=" --max-seq-length=512"
CMD+=" --max-predictions-per-seq=80"
CMD+=" --max-steps=$train_steps_phase2"
CMD+=" --warmup-proportion=$warmup_proportion_phase2"
CMD+=" --num-steps-per-checkpoint=$save_checkpoint_steps"
CMD+=" --learning-rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" --log-freq=1"
CMD+=" --optimizer=Lamb"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $BENCH"
CMD+=" --from-pretrained-params=${PHASE1_END_CKPT_DIR} "
CMD+=" --phase2 "
CMD+=" --report-file ${RESULTS_DIR}/dllogger_p2.json "

CMD="$DIST_CMD $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus \* $gradient_accumulation_steps_phase2)
  printf -v TAG "paddle_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
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
