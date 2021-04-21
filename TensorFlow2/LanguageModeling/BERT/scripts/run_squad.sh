#!/usr/bin/env bash

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

num_gpu=${1:-"8"}
batch_size=${2:-"8"}
learning_rate=${3:-"5e-6"}
precision=${4:-"fp16"}
use_xla=${5:-"true"}
bert_model=${6:-"large"}
squad_version=${7:-"1.1"}
epochs=${8:-"2"}

if [ $num_gpu -gt 1 ] ; then
    mpi_command="mpirun -np $num_gpu \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib"
    use_hvd="--use_horovod"
else
    mpi_command=""
    use_hvd=""
fi

if [ "$precision" = "fp16" ] ; then
    echo "fp16 activated!"
    use_fp16="--use_fp16"
else
    use_fp16=""
fi

if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--enable_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi

if [ "$bert_model" = "large" ] ; then
    export BERT_BASE_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    export BERT_BASE_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
fi

export SQUAD_VERSION=v$squad_version
export SQUAD_DIR=data/download/squad/$SQUAD_VERSION

export GBS=$(expr $batch_size \* $num_gpu)
printf -v TAG "tf_bert_finetuning_squad_%s_%s_gbs%d" "$bert_model" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results/${TAG}_${DATESTAMP}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

set -x
$mpi_command python run_squad.py \
  --mode=train_and_predict \
  --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
  --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
  --predict_file=${SQUAD_DIR}/dev-${SQUAD_VERSION}.json \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=$batch_size \
  --learning_rate=$learning_rate \
  --num_train_epochs=$epochs \
  --model_dir=${RESULTS_DIR} \
  --eval_script=$SQUAD_DIR/evaluate-$SQUAD_VERSION.py \
  $use_hvd $use_fp16 $use_xla_tag |& tee $LOGFILE

set +x
