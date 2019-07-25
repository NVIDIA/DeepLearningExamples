#!/usr/bin/env bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

batch_size=${1:-"8"}
learning_rate=${2:-"5e-6"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}
num_gpu=${5:-"8"}
seq_length=${6:-"384"}
doc_stride=${7:-"128"}
bert_model=${8:-"large"}

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=data/pretrained_models_google/uncased_L-24_H-1024_A-16
else
    export BERT_DIR=data/pretrained_models_google/uncased_L-12_H-768_A-12
fi

squad_version=${9:-"1.1"}

export SQUAD_DIR=data/squad/v${squad_version}
if [ "$squad_version" = "1.1" ] ; then
    version_2_with_negative="False"
else
    version_2_with_negative="True"
fi

init_checkpoint=${10:-"$BERT_DIR/bert_model.ckpt"}
epochs=${11:-"2.0"}

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

if [ $num_gpu -gt 1 ] ; then
    mpi_command="mpirun -np $num_gpu -H localhost:$num_gpu \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib"
    use_hvd="--horovod"
else
    mpi_command=""
    use_hvd=""
fi


  export GBS=$(expr $batch_size \* $num_gpu)
  printf -v TAG "tf_bert_%s_squad_1n_%s_gbs%d" "$bert_model" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`

  RESULTS_DIR=${RESULTS_DIR}/${TAG}_${DATESTAMP}
  mkdir $RESULTS_DIR
  LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
  printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
  printf "Writing logs to %s\n" "$LOGFILE"

    $mpi_command python run_squad.py \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$init_checkpoint \
    --do_train=True \
    --train_file=$SQUAD_DIR/train-v${squad_version}.json \
    --do_predict=True \
    --predict_file=$SQUAD_DIR/dev-v${squad_version}.json \
    --train_batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --num_train_epochs=$epochs \
    --max_seq_length=$seq_length \
    --doc_stride=$doc_stride \
    --save_checkpoints_steps 1000 \
    --output_dir=$RESULTS_DIR \
    "$use_hvd" \
    "$use_fp16" \
    $use_xla_tag --version_2_with_negative=${version_2_with_negative} |& tee $LOGFILE

python $SQUAD_DIR/evaluate-v${squad_version}.py $SQUAD_DIR/dev-v${squad_version}.json ${RESULTS_DIR}/predictions.json |& tee -a $LOGFILE
