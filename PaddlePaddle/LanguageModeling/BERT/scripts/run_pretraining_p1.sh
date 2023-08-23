# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

python3 -m paddle.distributed.launch \
--gpus="0,1,2,3,4,5,6,7" \
./run_pretraining.py \
--input-dir=pretrain/phase1/unbinned/parquet \
--vocab-file=vocab/bert-large-uncased-vocab.txt \
--output-dir=./results/checkpoints \
--bert-model=bert-large-uncased \
--from-checkpoint=./results/checkpoints/bert-large-uncased/phase1 \
--last-step-of-checkpoint=auto \
--batch-size=256 \
--max-steps=7038 \
--num-steps-per-checkpoint=200 \
--log-freq=1 \
--max-seq-length=128 \
--max-predictions-per-seq=20 \
--gradient-merge-steps=32 \
--amp \
--use-dynamic-loss-scaling \
--optimizer=Lamb \
--fuse-mha \
--phase1 \
--scale-loss=1048576 \
--learning-rate=6e-3 \
--warmup-proportion=0.2843 \
--report-file=./results/dllogger_p1.json
