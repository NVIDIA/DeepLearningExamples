#!/bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

init_checkpoint=${1:-"/results/biobert_tf_uncased_base/model.ckpt-4340"}
train_batch_size=${2:-8}
learning_rate=${3:-1.5e-6}
cased=${4:-false}
precision=${5:-"fp16"}
use_xla=${6:-"true"}
num_gpu=${7:-"16"}
seq_length=${8:-512}
bert_model=${9:-"base"}
eval_batch_size=${10:-16} #Eval and Predict BS is assumed to be same
epochs=${11:-"3.0"}

if [ "$cased" = "true" ] ; then
    DO_LOWER_CASE=0
    CASING_DIR_PREFIX="cased"
    case_flag="--do_lower_case=False"
else
    DO_LOWER_CASE=1
    CASING_DIR_PREFIX="uncased"
    case_flag="--do_lower_case=True"
fi

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=/workspace/bert/data/download/google_pretrained_weights/${CASING_DIR_PREFIX}_L-24_H-1024_A-16
else
    export BERT_DIR=/workspace/bert/data/download/google_pretrained_weights/${CASING_DIR_PREFIX}_L-12_H-768_A-12
fi

export GBS=$(expr $train_batch_size \* $num_gpu)
printf -v TAG "tf_bert_biobert_rel_chemprot_%s_%s_gbs%d" "$bert_model" "$precision" $GBS
DATESTAMP=`date +'%y%m%d%H%M%S'`


DATASET_DIR=/workspace/bert/data/biobert/chemprot-data_treeLSTM
OUTPUT_DIR=/results/${TAG}_${DATESTAMP}
mkdir -p ${OUTPUT_DIR}

use_fp16=""
if [ "$precision" = "fp16" ] ; then
        echo "fp16 activated!"
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

$mpi_command python3 /workspace/bert/run_re.py \
  --do_prepare=true \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --task_name="chemprot" \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$init_checkpoint \
  --num_train_epochs=$epochs \
  --data_dir=$DATASET_DIR \
  --output_dir=$OUTPUT_DIR \
  --learning_rate=$learning_rate \
  --train_batch_size=$train_batch_size \
  --eval_batch_size=$eval_batch_size \
  --predict_batch_size=$eval_batch_size \
  --max_seq_length=$seq_length \
  "$use_hvd" "$use_fp16" $use_xla_tag $case_flag

python3 /workspace/bert/biobert/re_eval.py --task=chemprot --output_path=$OUTPUT_DIR/test_results.tsv \
  --answer_path=$DATASET_DIR/test.tsv |& tee $OUTPUT_DIR/test_results.txt
