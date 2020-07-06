#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script does Quantization aware training of Resnet-50 by finetuning on the pre-trained model using 1 GPU and a batch size of 32.
# Usage ./GPU1_RN50_QAT.sh <path to the pre-trained model> <path to dataset> <path to results directory>

python main.py --mode=train_and_evaluate --batch_size=32 --lr_warmup_epochs=1 --quantize --symmetric --use_qdq --label_smoothing 0.1 --lr_init=0.00005 --momentum=0.875 --weight_decay=3.0517578125e-05 --finetune_checkpoint=$1 --data_dir=$2 --results_dir=$3 --num_iter 10 --data_format NHWC
