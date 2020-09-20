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

task=${1:-"squad"}

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=/results
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
echo "Results directory set as " $RESULTS_DIR

LOGFILE="${RESULTS_DIR}/${task}_inference_benchmark_bert_${bert_model}.log"
tmp_file="/tmp/${task}_inference_benchmark.log"
if [ "$task" = "squad" ] ; then
    export SQUAD_DIR=data/download/squad/v1.1

    echo "Squad directory set as " $SQUAD_DIR

    echo "Inference performance benchmarking for BERT $bert_model from $BERT_DIR" >> $LOGFILE
    for bert_model in "base" "large"; do
      echo "Model Sequence-Length Batch-size Precision Throughput-Average(sent/sec) Latency-Average(ms) Latency-50%(ms) Latency-90%(ms) Latency-95%(ms) Latency-99%(ms) Latency-100%(ms)" >> $LOGFILE


      if [ "$bert_model" = "large" ] ; then
        export BERT_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
      else
          export BERT_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
      fi
      echo  "BERT directory set as " $BERT_DIR

      init_checkpoint="$BERT_DIR/bert_model.ckpt"

      for seq_len in 128 384; do

      for bs in 1 2 4 8; do

      for use_fp16 in "--amp" "--noamp"; do

        python run_squad.py \
        --vocab_file=$BERT_DIR/vocab.txt \
        --bert_config_file=$BERT_DIR/bert_config.json \
        --init_checkpoint=$init_checkpoint \
        --do_predict=True \
        --predict_file=$SQUAD_DIR/dev-v1.1.json \
        --predict_batch_size=$bs \
        --max_seq_length=$seq_len \
        --doc_stride=128 \
        --output_dir=${RESULTS_DIR} \
        "$use_fp16" \
        --use_xla --num_eval_iterations=1024 |& tee $tmp_file

        perf=`cat $tmp_file | grep -F 'INFO:tensorflow:Throughput Average (sentences/sec) =' | tail -1 | awk -F'= ' '{print $2}'`
        la=`cat $tmp_file | grep -F 'INFO:tensorflow:Latency Average (ms)' | awk -F'= ' '{print $2}'`
        l50=`cat $tmp_file | grep -F 'INFO:tensorflow:Latency Confidence Level 50 (ms)' | awk -F'= ' '{print $2}'`
        l90=`cat $tmp_file | grep -F 'INFO:tensorflow:Latency Confidence Level 90 (ms)' | awk -F'= ' '{print $2}'`
        l95=`cat $tmp_file | grep -F 'INFO:tensorflow:Latency Confidence Level 95 (ms)' | awk -F'= ' '{print $2}'`
        l99=`cat $tmp_file | grep -F 'INFO:tensorflow:Latency Confidence Level 99 (ms)' | awk -F'= ' '{print $2}'`
        l100=`cat $tmp_file | grep -F 'INFO:tensorflow:Latency Confidence Level 100 (ms)' | awk -F'= ' '{print $2}'`

        echo "$bert_model $seq_len $bs $use_fp16 $perf $la $l50 $l90 $l95 $l99 $l100" >> $LOGFILE

       done
       done
       done
    done

else

    echo "Benchmarking for " $task "currently not supported. Sorry!"

fi