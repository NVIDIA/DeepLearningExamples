# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

: ${SEED:=1}
: ${LR:=1e-3}
: ${NGPU:=8}
: ${BATCH_SIZE:=1024}
: ${EPOCHS:=30}

python -m torch.distributed.run --nproc_per_node=${NGPU} train.py \
        --dataset electricity \
        --data_path /data/processed/electricity_bin \
        --batch_size=${BATCH_SIZE} \
        --sample 450000 50000 \
        --lr ${LR} \
        --epochs ${EPOCHS} \
        --seed ${SEED} \
        --use_amp \
        --results /results/TFT_electricity_bs${NGPU}x${BATCH_SIZE}_lr${LR}/seed_${SEED}
