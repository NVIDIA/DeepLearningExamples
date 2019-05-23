#!/usr/bin/env bash

export SQUAD_DIR=data/squad/v1.1
export BERT_DIR=data/pretrained_models_google/uncased_L-24_H-1024_A-16

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${1:-"/results/model.ckpt"}
batch_size=${2:-"8"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}

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

python run_squad.py \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$init_checkpoint \
--do_predict=True \
--predict_file=$SQUAD_DIR/dev-v1.1.json \
--max_seq_length=384 \
--doc_stride=128 \
--predict_batch_size=$batch_size \
--output_dir=/results \
"$use_fp16" \
$use_xla_tag

python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /results/predictions.json
