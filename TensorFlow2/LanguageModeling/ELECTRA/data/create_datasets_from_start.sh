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

to_download=${1:-"wiki_only"}

#Download
if [ "$to_download" = "wiki_books" ] ; then
    python3 /workspace/electra/data/dataPrep.py --action download --dataset bookscorpus
fi
python3 /workspace/electra/data/dataPrep.py --action download --dataset wikicorpus_en

#Download SQuAD
python3 /workspace/electra/data/dataPrep.py --action download --dataset squad

# Properly format the text files
if [ "$to_download" = "wiki_books" ] ; then
    python3 /workspace/electra/data/dataPrep.py --action text_formatting --dataset bookscorpus
fi
python3 /workspace/electra/data/dataPrep.py --action text_formatting --dataset wikicorpus_en

if [ "$to_download" = "wiki_books" ] ; then
    DATASET="books_wiki_en_corpus"
else
    DATASET="wikicorpus_en"
    # Shard the text files
fi

# Shard the text files (group wiki+books then shard)
python3 /workspace/electra/data/dataPrep.py --action sharding --dataset $DATASET --n_test_shards 2048 --n_training_shards 2048

# Create tfrecoreds files Phase 1
python3 /workspace/electra/data/dataPrep.py --action create_tfrecord_files --dataset $DATASET --max_seq_length 128 --n_test_shards 2048 --n_training_shards 2048 --vocab_file=vocab/vocab.txt --do_lower_case=1

# Create tfrecoreds files Phase 2
python3 /workspace/electra/data/dataPrep.py --action create_tfrecord_files --dataset $DATASET --max_seq_length 512 --n_test_shards 2048 --n_training_shards 2048 --vocab_file=vocab/vocab.txt --do_lower_case=1
