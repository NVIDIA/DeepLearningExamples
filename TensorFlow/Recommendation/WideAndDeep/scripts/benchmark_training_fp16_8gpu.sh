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

set -x
set -e

mpiexec --allow-run-as-root --bind-to socket -np 8 python -m trainer.task --hvd --model_dir . --transformed_metadata_path "/outbrain/tfrecords" --eval_data_pattern "/outbrain/tfrecords/eval_*" --train_data_pattern "/outbrain/tfrecords/train_*" --save_checkpoints_secs 600 --linear_l1_regularization 0.0 --linear_l2_regularization 0.0 --linear_learning_rate 0.2 --deep_l1_regularization 0.0 --deep_l2_regularization 0.0 --deep_learning_rate 1.0 --deep_dropout 0.0 --deep_hidden_units 1024 1024 1024 1024 1024 --prebatch_size 4096 --global_batch_size 131072 --eval_batch_size 32768 --eval_steps 8 --model_type wide_n_deep --gpu --benchmark --amp
