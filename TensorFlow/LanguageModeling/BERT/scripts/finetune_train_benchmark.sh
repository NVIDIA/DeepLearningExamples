#!/bin/bash

export BERT_DIR=data/pretrained_models_google/uncased_L-24_H-1024_A-16

task=${1:-"squad"}
precision=${2:-"fp16"}
use_xla=${3:-"true"}
num_gpu=${4:-"8"}
batch_size=${5:-"8"}
learning_rate=${6:-"5e-6"}

if [ "$task" = "squad" ] ; then
    export SQUAD_DIR=data/squad/v1.1

    epochs="2.0"
    use_fp16=""
    LOGFILE="/results/${task}_training_benchmark.log"
    if [ "$precision" = "fp16" ] ; then
            export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
            use_fp16="--use_fp16"
    fi


    if [ "$use_xla" = "true" ] ; then
        use_xla_tag="--use_xla"
    else
        use_xla_tag=""
    fi

    if [ "$num_gpu" = "0" ] ; then
        mpi_command=""
        use_hvd=""
    else
        mpi_command="mpirun -np $num_gpu -H localhost:$num_gpu \
        --allow-run-as-root -bind-to none -map-by slot \
        -x NCCL_DEBUG=INFO \
        -x LD_LIBRARY_PATH \
        -x PATH -mca pml ob1 -mca btl ^openib"
        use_hvd="--horovod"
    fi


    $mpi_command python run_squad.py \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/bert_model.ckpt \
    --do_train=True \
    --train_file=$SQUAD_DIR/train-v1.1.json \
    --train_batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --num_train_epochs=$epochs \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=/results \
    "$use_hvd" \
    "$use_fp16" \
    $use_xla_tag &> $LOGFILE

    perf=`cat $LOGFILE | grep -F 'Training Performance' | awk -F'= ' '{print $2}'`
    echo "Training performance is $perf"

else

    echo "Benchmarking for " $task "currently not supported. Sorry!"

fi