#!/usr/bin/env bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

wiki_path=${1:-"/workspace/bart/data/wiki"}
common_crawl_path=${2:-"/workspace/bart/data/common_crawl"}
openwebtext_path=${3:-"/workspace/bart/data/openwebtext"}
output_path=${4:-"/workspace/bart/data"}

mpirun \
  -np 16 \
  --oversubscribe \
  --allow-run-as-root \
  preprocess_bart_pretrain \
  --schedule mpi \
      --target-seq-length 128 \
      --wikipedia $wiki_path/source \
      --common-crawl $common_crawl_path/source \
      --open-webtext $openwebtext_path/source \
      --sink $output_path/pretrain_lddl_128 \
      --num-blocks 1280

mpirun -np 16 --oversubscribe --allow-run-as-root \
  balance_dask_output \
    --indir $output_path/pretrain_lddl_128 \
    --num-shards 1280

mpirun \
  -np 16 \
  --oversubscribe \
  --allow-run-as-root \
  preprocess_bart_pretrain \
  --schedule mpi \
      --target-seq-length 512 \
      --wikipedia $wiki_path/source \
      --common-crawl $common_crawl_path/source \
      --open-webtext $openwebtext_path/source \
      --sink $output_path/pretrain_lddl_512 \
      --num-blocks 1280

mpirun -np 16 --oversubscribe --allow-run-as-root \
  balance_dask_output \
    --indir $output_path/pretrain_lddl_512 \
    --num-shards 1280
