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

init_checkpoint=${1:-"/results/model.ckpt"}
batch_size=${2:-"8"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}
bert_model=${5:-"large"}
squad_version=${6:-"1.1"}

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
fi

export SQUAD_DIR=data/download/squad/v${squad_version}
export SQUAD_VERSION=v$squad_version

echo "Squad directory set as " $SQUAD_DIR " BERT directory set as " $BERT_DIR
echo "Results directory set as " $RESULTS_DIR

use_fp16=""
if [ "$precision" = "fp16" ] ; then
        echo "fp16 activated!"
        use_fp16="--use_fp16"
fi

if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--enable_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi

ckpt_str=${init_checkpoint//\//-}
printf -v TAG "tf_bert_finetuning_squad_%s_inf_%s_gbs%d_ckpt_%s" "$bert_model" "$precision" $batch_size "$ckpt_str"
DATESTAMP=`date +'%y%m%d%H%M%S'`
#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"
mkdir -p $RESULTS_DIR

#Check if all necessary files are available before training
for DIR_or_file in $SQUAD_DIR $RESULTS_DIR $BERT_DIR/vocab.txt $BERT_DIR/bert_config.json; do
  if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
     echo "Error! $DIR_or_file directory missing. Please mount correctly"
     exit -1
  fi
done

python run_squad.py \
--mode=predict \
--input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$init_checkpoint \
--predict_file=$SQUAD_DIR/dev-v${squad_version}.json \
--predict_batch_size=$batch_size \
--model_dir=$RESULTS_DIR \
$use_fp16 $use_xla_tag

python $SQUAD_DIR/evaluate-v${squad_version}.py $SQUAD_DIR/dev-v${squad_version}.json $RESULTS_DIR/predictions.json
