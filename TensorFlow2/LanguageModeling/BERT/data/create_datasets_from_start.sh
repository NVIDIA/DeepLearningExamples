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

export BERT_PREP_WORKING_DIR=/workspace/bert_tf2/data

to_download=${1:-"all"}
pretrained_to_download=${2:-"wiki_only"} # By default, we don't download BooksCorpus dataset due to recent issues with the host server

if [ "$to_download" = "all" ] || [ "$to_download" = "squad" ] ; then
    #SQUAD
    python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action download --dataset google_pretrained_weights  # Includes vocab

    python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action download --dataset squad

    export BERT_DIR=${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
    export SQUAD_DIR=${BERT_PREP_WORKING_DIR}/download/squad
    python create_finetuning_data.py \
    --squad_data_file=${SQUAD_DIR}/v1.1/train-v1.1.json \
    --vocab_file=${BERT_DIR}/vocab.txt \
    --train_data_output_path=${SQUAD_DIR}/v1.1/squad_v1.1_train.tf_record \
    --meta_data_file_path=${SQUAD_DIR}/v1.1/squad_v1.1_meta_data \
    --fine_tuning_task_type=squad --max_seq_length=384

    python create_finetuning_data.py \
    --squad_data_file=${SQUAD_DIR}/v2.0/train-v2.0.json \
    --vocab_file=${BERT_DIR}/vocab.txt \
    --train_data_output_path=${SQUAD_DIR}/v2.0/squad_v2.0_train.tf_record \
    --meta_data_file_path=${SQUAD_DIR}/v2.0/squad_v2.0_meta_data \
    --fine_tuning_task_type=squad --max_seq_length=384 --version_2_with_negative=True
fi

if [ "$to_download" = "all" ] || [ "$to_download" = "pretrained" ] ; then
    #Pretrained
    if [ "$pretrained_to_download" = "wiki_books" ] ; then
        python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action download --dataset bookscorpus
    fi

    python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action download --dataset wikicorpus_en

    DATASET="wikicorpus_en"
    # Properly format the text files
    if [ "$pretrained_to_download" = "wiki_books" ] ; then
        python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action text_formatting --dataset bookscorpus
        DATASET="books_wiki_en_corpus"
    fi
    python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action text_formatting --dataset wikicorpus_en

    # Shard the text files
    python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action sharding --dataset $DATASET

    # Create TFRecord files Phase 1
    python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action create_tfrecord_files --dataset ${DATASET} --max_seq_length 128 \
    --max_predictions_per_seq 20 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt


    # Create TFRecord files Phase 2
    python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action create_tfrecord_files --dataset ${DATASET} --max_seq_length 512 \
    --max_predictions_per_seq 80 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt
fi