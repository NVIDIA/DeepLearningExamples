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

MODEL_DIR="./output"
DATA_DIR="/data"
INDX="./index_file"

# TF_XLA_FLAGS=--tf_xla_cpu_global_jit

horovodrun -np 8 bash ./scripts/bind.sh --cpu=exclusive --ib=single -- python3 main.py \
  --mode "train_and_eval" \
  --arch "efficientnet-b4" \
  --model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --use_amp \
  --use_xla \
  --augmenter_name autoaugment \
  --weight_init fan_out \
  --lr_decay cosine \
  --max_epochs 2 \
  --train_batch_size 160 \
  --eval_batch_size 160 \
  --log_steps 100 \
  --save_checkpoint_freq 5 \
  --lr_init 0.005 \
  --batch_norm syncbn \
  --mixup_alpha 0.2 \
  --weight_decay 5e-6 \
  --resume_checkpoint
