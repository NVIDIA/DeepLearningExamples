#!/usr/bin/env bash

export SQUAD_DIR=data/squad/v1.1
export BERT_DIR=data/pretrained_models_google/uncased_L-24_H-1024_A-16


echo "Container nvidia build = " $NVIDIA_BUILD_ID

batch_size=${1:-"8"}
learning_rate=${2:-"5e-6"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}
num_gpu=${5:-"8"}
init_checkpoint=${6:-"$BERT_DIR/bert_model.ckpt"}
epochs="2.0"

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
    --init_checkpoint=$init_checkpoint \
    --do_train=True \
    --train_file=$SQUAD_DIR/train-v1.1.json \
    --do_predict=True \
    --predict_file=$SQUAD_DIR/dev-v1.1.json \
    --train_batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --num_train_epochs=$epochs \
    --max_seq_length=384 \
    --doc_stride=128 \
    --save_checkpoints_steps 1000 \
    --output_dir=/results \
    "$use_hvd" \
    "$use_fp16" \
    $use_xla_tag 

python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /results/predictions.json
