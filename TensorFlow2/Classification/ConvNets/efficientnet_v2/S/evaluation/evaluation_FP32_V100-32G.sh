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

python main.py \
        --cfg config/efficientnet_v2/s_cfg.py \
        --mode eval \
        --use_xla \
        --eval_batch_size 128 \
        --eval_img_size 384 \
        --model_dir ./output/expXX \
        --n_repeat_eval 4 \
        --moving_average_decay 0.9999 # enables evaluation using EMA weights too
