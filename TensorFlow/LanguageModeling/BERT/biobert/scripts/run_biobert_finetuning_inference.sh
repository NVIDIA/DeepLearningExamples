#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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
task=${1:-"ner_bc5cdr-chem"}
init_checkpoint=${2:-"/results/biobert_tf_uncased_base/model.ckpt-4340"}
bert_model=${3:-"base"}
cased=${4:-"false"}
precision=${5:-"fp16"}
use_xla=${6:-"true"}
batch_size=${7:-"16"}

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

DATESTAMP=`date +'%y%m%d%H%M%S'`

if [ "$task" = "ner_bc5cdr-chem" ] ; then

  printf -v TAG "tf_bert_biobert_ner_bc5cdr_chem_inference_%s_%s" "$bert_model" "$precision"
  DATASET_DIR=/workspace/bert/data/biobert/BC5CDR/chem
  OUTPUT_DIR=/results/${TAG}_${DATESTAMP}

  python /workspace/bert/run_ner.py \
  --do_prepare=true \
  --do_eval=true \
  --do_predict=true \
  --task_name="bc5cdr" \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$init_checkpoint \
  --data_dir=$DATASET_DIR \
  --output_dir=$OUTPUT_DIR \
  --eval_batch_size=$batch_size \
  --predict_batch_size=$batch_size \
  --max_seq_length=128 \
  $use_fp16 $use_xla_tag $case_flag

elif [ "$task" = "ner_bc5cdr-disease" ] ; then
  printf -v TAG "tf_bert_biobert_ner_bc5cdr_disease_inference_%s_%s" "$bert_model" "$precision"
  DATASET_DIR=/workspace/bert/data/biobert/BC5CDR/disease
  OUTPUT_DIR=/results/${TAG}_${DATESTAMP}

  python3 /workspace/bert/run_ner.py \
  --do_prepare=true \
  --do_eval=true \
  --do_predict=true \
  --task_name="bc5cdr" \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$init_checkpoint \
  --data_dir=$DATASET_DIR \
  --output_dir=$OUTPUT_DIR \
  --eval_batch_size=$batch_size \
  --predict_batch_size=$batch_size \
  --max_seq_length=128 \
  "$use_fp16" $use_xla_tag $case_flag

elif [ "$task" = "rel_chemprot" ] ; then
  printf -v TAG "tf_bert_biobert_rel_chemprot_inference_%s_%s_" "$bert_model" "$precision"
  DATASET_DIR=/workspace/bert/data/biobert/chemprot-data_treeLSTM
  OUTPUT_DIR=/results/${TAG}_${DATESTAMP}

  python3 /workspace/bert/run_re.py \
  --do_prepare=true \
  --do_eval=true \
  --do_predict=true \
  --task_name="chemprot" \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$init_checkpoint \
  --data_dir=$DATASET_DIR \
  --output_dir=$OUTPUT_DIR \
  --eval_batch_size=$batch_size \
  --predict_batch_size=$batch_size \
  --max_seq_length=512 \
  "$use_fp16" $use_xla_tag $case_flag

  python3 /workspace/bert/biobert/re_eval.py --task=chemprot --output_path=$OUTPUT_DIR/test_results.tsv \
  --answer_path=$DATASET_DIR/test.tsv |& tee $OUTPUT_DIR/test_results.txt

else

    echo "Benchmarking for " $task "currently not supported. Sorry!"

fi