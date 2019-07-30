#!/usr/bin/env bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${1:-"/results/model.ckpt"}
batch_size=${2:-"8"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}
seq_length=${5:-"384"}
doc_stride=${6:-"128"}
bert_model=${7:-"large"}
squad_version=${8:-"1.1"}

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=data/pretrained_models_google/uncased_L-24_H-1024_A-16
else
    export BERT_DIR=data/pretrained_models_google/uncased_L-12_H-768_A-12
fi

export SQUAD_DIR=data/squad/v${squad_version}
if [ "$squad_version" = "1.1" ] ; then
    version_2_with_negative="False"
else
    version_2_with_negative="True"
fi

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results

if [ ! -d "$SQUAD_DIR" ] ; then
   echo "Error! $SQUAD_DIR directory missing. Please mount SQuAD dataset."
   exit -1
fi
if [ ! -d "$BERT_DIR" ] ; then
   echo "Error! $BERT_DIR directory missing. Please mount pretrained BERT dataset."
   exit -1
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi

echo "Squad directory set as " $SQUAD_DIR " BERT directory set as " $BERT_DIR
echo "Results directory set as " $RESULTS_DIR

use_fp16=""
if [ "$precision" = "fp16" ] ; then
        echo "fp16 activated!"
        export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
        use_fp16="--use_fp16"
fi

if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--use_xla"
    echo "XLA activated"
else
    use_xla_tag=""
fi

  printf -v TAG "tf_bert_%s_squad_inf_1n_%s_gbs%d_ckpt_%s" "$bert_model" "$precision" $batch_size "$init_checkpoint"
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
  printf "Writing logs to %s\n" "$LOGFILE"

python run_squad.py \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$init_checkpoint \
--do_predict=True \
--predict_file=$SQUAD_DIR/dev-v${squad_version}.json \
--max_seq_length=$seq_length \
--doc_stride=$doc_stride \
--predict_batch_size=$batch_size \
--output_dir=$RESULTS_DIR \
"$use_fp16" \
$use_xla_tag --version_2_with_negative=${version_2_with_negative}

python $SQUAD_DIR/evaluate-v${squad_version}.py $SQUAD_DIR/dev-v${squad_version}.json $RESULTS_DIR/predictions.json
