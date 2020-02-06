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
bert_model=${2:-"base"}
cased=${3:-"false"}

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

DATESTAMP=`date +'%y%m%d%H%M%S'`
printf -v TAG "tf_bert_biobert_%s_inference_benchmark_%s_%s" "$task" "$bert_model" "$CASING_DIR_PREFIX"
OUTPUT_DIR=/results/${TAG}_${DATESTAMP}
mkdir -p ${OUTPUT_DIR}

if [ "$task" = "ner_bc5cdr-chem" ] ; then


  DATASET_DIR=/workspace/bert/data/biobert/BC5CDR/chem
  LOGFILE="${OUTPUT_DIR}/${task}_training_benchmark_bert_${bert_model}.log"

    echo "Training performance benchmarking for BERT $bert_model from $BERT_DIR" >> $LOGFILE

    echo "Precision Sequence Length   Batch size  Performance(sent/sec)" >> $LOGFILE

    for seq_length in 128 512; do
        for batch_size in 8 32 64; do
            for precision in fp16 fp32; do
                res_dir=${OUTPUT_DIR}/bert_${bert_model}_sl_${seq_len}_prec_${precision}_bs_${batch_size}
                mkdir -p ${res_dir}
                tmp_file="${res_dir}/${task}_training_benchmark.log"

                if [ "$precision" = "fp16" ] ; then
                    echo "fp16 activated!"
                    use_fp16="--use_fp16"
                    use_xla_tag="--use_xla"
                else
                    echo "fp32 activated!"
                    use_fp16=""
                    use_xla_tag=""
                fi

                python /workspace/bert/run_ner.py \
                --do_prepare=true \
                --do_eval=true \
                --do_predict=true \
                --task_name="bc5cdr" \
                --vocab_file=$BERT_DIR/vocab.txt \
                --bert_config_file=$BERT_DIR/bert_config.json \
                --init_checkpoint="$BERT_DIR/bert_model.ckpt" \
                --data_dir=$DATASET_DIR \
                --output_dir=$res_dir \
                --eval_batch_size=$batch_size \
                --predict_batch_size=$batch_size \
                --max_seq_length=$seq_length \
                $use_fp16 $use_xla_tag $case_flag  |& tee $tmp_file

                  perf=`cat $tmp_file | grep -F 'Throughput Average (sentences/sec) =' | tail -1 | awk -F'= ' '{print $2}' | awk -F' sen' '{print $1}'`
                echo "$precision $seq_len  $batch_size $perf" >> $LOGFILE

            done
        done
    done

elif [ "$task" = "ner_bc5cdr-disease" ] ; then
  DATASET_DIR=/workspace/bert/data/biobert/BC5CDR/disease

  LOGFILE="${OUTPUT_DIR}/${task}_training_benchmark_bert_${bert_model}.log"

    echo "Training performance benchmarking for BERT $bert_model from $BERT_DIR" >> $LOGFILE

    echo "Precision Sequence Length   Batch size  Performance(sent/sec)" >> $LOGFILE

    for seq_length in 128 512; do
        for batch_size in 8 32 64; do
            for precision in fp16 fp32; do
                res_dir=${OUTPUT_DIR}/bert_${bert_model}_sl_${seq_len}_prec_${precision}_bs_${batch_size}
                mkdir -p ${res_dir}
                tmp_file="${res_dir}/${task}_training_benchmark.log"

                if [ "$precision" = "fp16" ] ; then
                    echo "fp16 activated!"
                    use_fp16="--use_fp16"
                    use_xla_tag="--use_xla"
                else
                    echo "fp32 activated!"
                    use_fp16=""
                    use_xla_tag=""
                fi
                python3 /workspace/bert/run_ner.py \
                --do_prepare=true \
                --do_eval=true \
                --do_predict=true \
                --task_name="bc5cdr" \
                --vocab_file=$BERT_DIR/vocab.txt \
                --bert_config_file=$BERT_DIR/bert_config.json \
                --init_checkpoint="$BERT_DIR/bert_model.ckpt" \
                --data_dir=$DATASET_DIR \
                --output_dir=$res_dir \
                --eval_batch_size=$batch_size \
                --predict_batch_size=$batch_size \
                --max_seq_length=$seq_length \
                "$use_fp16" $use_xla_tag $case_flag  |& tee $tmp_file

                  perf=`cat $tmp_file | grep -F 'Throughput Average (sentences/sec) =' | tail -1 | awk -F'= ' '{print $2}' | awk -F' sen' '{print $1}'`
                echo "$precision $seq_len  $batch_size $perf" >> $LOGFILE

            done
        done
    done

elif [ "$task" = "rel_chemprot" ] ; then
  DATASET_DIR=/workspace/bert/data/biobert/chemprot-data_treeLSTM

  LOGFILE="${OUTPUT_DIR}/${task}_training_benchmark_bert_${bert_model}.log"

    echo "Training performance benchmarking for BERT $bert_model from $BERT_DIR" >> $LOGFILE

    echo "Precision Sequence Length   Batch size  Performance(sent/sec)" >> $LOGFILE

    for seq_length in 128 512; do
        for batch_size in 8 32 64; do
            for precision in fp16 fp32; do
                res_dir=${OUTPUT_DIR}/bert_${bert_model}_sl_${seq_len}_prec_${precision}_bs_${batch_size}
                mkdir -p ${res_dir}
                tmp_file="${res_dir}/${task}_training_benchmark.log"

                if [ "$precision" = "fp16" ] ; then
                    echo "fp16 activated!"
                    use_fp16="--use_fp16"
                    use_xla_tag="--use_xla"
                else
                    echo "fp32 activated!"
                    use_fp16=""
                    use_xla_tag=""
                fi
                python3 /workspace/bert/run_re.py \
                --do_prepare=true \
                --do_eval=true \
                --do_predict=true \
                --task_name="chemprot" \
                --vocab_file=$BERT_DIR/vocab.txt \
                --bert_config_file=$BERT_DIR/bert_config.json \
                --init_checkpoint="$BERT_DIR/bert_model.ckpt" \
                --data_dir=$DATASET_DIR \
                --output_dir=$res_dir \
                --eval_batch_size=$batch_size \
                --predict_batch_size=$batch_size \
                --max_seq_length=$seq_length \
                "$use_fp16" $use_xla_tag $case_flag  |& tee $tmp_file

                  perf=`cat $tmp_file | grep -F 'Throughput Average (sentences/sec) =' | tail -1 | awk -F'= ' '{print $2}' | awk -F' sen' '{print $1}'`
                echo "$precision $seq_len  $batch_size $perf" >> $LOGFILE

            done
        done
    done

else

    echo "Benchmarking for " $task "currently not supported. Sorry!"

fi