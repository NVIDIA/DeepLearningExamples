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

horovodrun -np 8 bash ./scripts/bind.sh --cpu=exclusive --ib=single -- python3 main.py \
        --cfg config/efficientnet_v1/b0_cfg.py \
        --mode train_and_eval \
        --use_xla \
        --model_dir ./output \
        --data_dir /data \
        --log_steps 100 \
        --max_epochs 3 \
        --save_checkpoint_freq 5 \
        --train_batch_size 512 \
        --eval_batch_size 512 \
        --augmenter_name autoaugment \
        --lr_decay cosine \
        --memory_limit 81000 \
        --defer_img_mixing \
        --moving_average_decay 0.9999 \
        --lr_init 0.005