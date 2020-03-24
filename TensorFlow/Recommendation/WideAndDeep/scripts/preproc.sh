#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
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

if [ $# -ge 1 ]
then
  PREBATCH_SIZE=$1
else
  PREBATCH_SIZE=4096
fi

python preproc/preproc1.py
python preproc/preproc2.py
python preproc/preproc3.py

export CUDA_VISIBLE_DEVICES=
LOCAL_DATA_DIR=/outbrain/preprocessed
LOCAL_DATA_TFRECORDS_DIR=/outbrain/tfrecords

TRAIN_DIR=train_feature_vectors_integral_eval.csv
VALID_DIR=validation_feature_vectors_integral.csv
TRAIN_IMPUTED_DIR=train_feature_vectors_integral_eval_imputed.csv
VALID_IMPUTED_DIR=validation_feature_vectors_integral_imputed.csv
HEADER_PATH=train_feature_vectors_integral_eval.csv.header

cd ${LOCAL_DATA_DIR}
python /wd/preproc/csv_data_imputation.py --num_workers 40 \
  --train_files_pattern 'train_feature_vectors_integral_eval.csv/part-*' \
  --valid_files_pattern 'validation_feature_vectors_integral.csv/part-*' \
  --train_dst_dir ${TRAIN_IMPUTED_DIR} \
  --valid_dst_dir ${VALID_IMPUTED_DIR} \
  --header_path ${HEADER_PATH}
cd -

time preproc/sort_csv.sh ${LOCAL_DATA_DIR}/${VALID_IMPUTED_DIR} ${LOCAL_DATA_DIR}/${VALID_IMPUTED_DIR}_sorted

python dataflow_preprocess.py \
  --eval_data "${LOCAL_DATA_DIR}/${VALID_IMPUTED_DIR}_sorted/part-*" \
  --training_data "${LOCAL_DATA_DIR}/${TRAIN_IMPUTED_DIR}/part-*" \
  --output_dir ${LOCAL_DATA_TFRECORDS_DIR} \
  --batch_size ${PREBATCH_SIZE}

