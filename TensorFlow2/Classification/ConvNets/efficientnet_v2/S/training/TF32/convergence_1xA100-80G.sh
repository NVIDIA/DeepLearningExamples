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

python3 main.py \
        --cfg config/efficientnet_v2/s_cfg.py \
        --mode train_and_eval \
        --use_xla \
        --model_dir ./output/ \
        --data_dir /data/ \
        --log_steps 500 \
        --save_checkpoint_freq 10 \
        --n_stages 4 \
        --max_epochs 350 \
        --train_batch_size 230 \
        --train_img_size 300 \
        --base_img_size 128 \
        --lr_decay cosine \
        --lr_init 0.005 \
        --weight_decay .000005 \
        --opt_epsilon 0.001 \
        --moving_average_decay 0.9999 \
        --eval_img_size 384 \
        --eval_batch_size 100 \
        --augmenter_name randaugment \
        --raug_num_layers 2 \
        --raug_magnitude 15 \
        --cutmix_alpha 0 \
        --mixup_alpha 0 \
        --defer_img_mixing 