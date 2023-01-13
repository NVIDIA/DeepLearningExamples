#!/usr/bin/env bash

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

set -eu

: ${DATASET_DIR:=/datasets/LibriSpeech}
: ${FILELISTS_DIR:=$DATASET_DIR}
: ${EXT:=flac}  # or wav

mkdir -p $DATASET_DIR
mkdir -p $FILELISTS_DIR

for SUBSET in train-clean-100 train-clean-360 train-other-500 \
              dev-clean dev-other test-clean test-other \
; do
    TSV=$FILELISTS_DIR/$SUBSET.tsv

    if [ ! -d $DATASET_DIR/$SUBSET ]; then
        echo "ERROR: $DATASET_DIR/$SUBSET does not exist; skipping."
        continue
    fi

    python3 utils/generate_filelist.py --extension $EXT $DATASET_DIR/$SUBSET $TSV
    python3 utils/libri_labels.py $TSV --output-dir $FILELISTS_DIR --output-name $SUBSET
done

# Combine
python3 utils/combine_filelists.py $FILELISTS_DIR/train-{clean-100,clean-360,other-500}.tsv > $FILELISTS_DIR/train-full-960.tsv

cat $FILELISTS_DIR/train-clean-100.wrd > $FILELISTS_DIR/train-full-960.wrd
cat $FILELISTS_DIR/train-clean-360.wrd >> $FILELISTS_DIR/train-full-960.wrd
cat $FILELISTS_DIR/train-other-500.wrd >> $FILELISTS_DIR/train-full-960.wrd

cat $FILELISTS_DIR/train-clean-100.ltr > $FILELISTS_DIR/train-full-960.ltr
cat $FILELISTS_DIR/train-clean-360.ltr >> $FILELISTS_DIR/train-full-960.ltr
cat $FILELISTS_DIR/train-other-500.ltr >> $FILELISTS_DIR/train-full-960.ltr

python3 utils/generate_dictionary.py $FILELISTS_DIR/train-full-960.ltr $FILELISTS_DIR/dict.ltr.txt
