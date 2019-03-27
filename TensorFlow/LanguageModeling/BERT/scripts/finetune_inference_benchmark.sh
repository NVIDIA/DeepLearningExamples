#!/bin/bash

export BERT_DIR=data/pretrained_models_google/uncased_L-24_H-1024_A-16

task=${1:-"squad"}
precision=${2:-"fp16"}
use_xla=${3:-"true"}
batch_size=${4:-"8"}
init_checkpoint=${5:-"$BERT_DIR/bert_model.ckpt"}


if [ "$task" = "squad" ] ; then
    export SQUAD_DIR=data/squad/v1.1

    LOGFILE="/results/${task}_inference_benchmark.log"

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
    --predict_batch_size=$batch_size \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=/results \
    "$use_fp16" \
    $use_xla_tag &> $LOGFILE

    perf=`cat $LOGFILE | grep -F 'Inference Performance' | awk -F'= ' '{print $2}'`
    echo "Inference performance is $perf"

else

    echo "Benchmarking for " $task "currently not supported. Sorry!"

fi