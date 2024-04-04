# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

set -x
set -e
: ${MODEL:=nbeats}
: ${DATASET:=electricity}
: ${ORDER:=3}
: ${SUFFIX:=best_0}
: ${RESULTS:='/results'}


RESULTS=${RESULTS}/${MODEL}_${DATASET}_${SUFFIX}_checkpoints

python launch_training.py \
    -m \
    seed="range(1,$(( 2 ** ${ORDER} + 1)))" \
    model=${MODEL} \
    dataset=${DATASET}            \
    overrides=${DATASET}/${MODEL}/${SUFFIX} \
    trainer.config.log_interval=-1 \
    ~trainer.callbacks.early_stopping \
    +trainer.config.force_rerun=True \
    evaluator.config.metrics=[MAE,RMSE,SMAPE,TDI] \
    hydra.sweep.dir=${RESULTS} \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs=8 \
    hydra.sweeper.max_batch_size=8

rm ${RESULTS}/*/last_checkpoint.zip

# Iterate over orders of magnitude
for J in $( seq 1 ${ORDER} )
do
    export WANDB_RUN_GROUP="${MODEL}_${DATASET}_ensembling_$(( 2 ** $J ))_${SUFFIX}"
    # For each order of magnitude split available checkpoint into 2^(order-J) disjoint sets and compute results on these to reduce variance
    for SHIFT in $( seq 0 $(( 2 ** $J )) $(( 2 ** ${ORDER} - 1)))
    do

        MODEL_LIST="["
        for I in $( seq 0 $(( 2 ** $J - 1)) )
        do
            MODEL_LIST+="{dir: ${RESULTS}/$(( $I + $SHIFT )), checkpoint: best_checkpoint.zip, weight:1.0},"
        done
        MODEL_LIST=${MODEL_LIST::-1}
        MODEL_LIST+="]"

        python launch_ensembling.py \
            model.config.model_list="${MODEL_LIST}"
    done
done
