# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
: ${SUFFIX:=}
: ${DISTRIBUTED:=0}

RESULTS=/results/${MODEL}_${DATASET}_hp_search${SUFFIX}
mkdir -p ${RESULTS}

if [[ ${DISTRIBUTED} == 0 ]]
then
    LAUNCHER='hydra.sweeper.experiment_sequence=hydra_utils.TSPPOptunaExperimentSequence '
    LAUNCHER+='hydra/launcher=multiprocessing '
    LAUNCHER+='hydra.launcher.n_jobs=8'
else
    LAUNCHER='hydra/launcher=torchrun '
fi

python launch_training.py \
    -m \
    model=${MODEL} \
    dataset=${DATASET}             \
    overrides=${DATASET}/${MODEL}/hp_search${SUFFIX} \
    trainer.config.log_interval=-1 \
    +trainer.config.force_rerun=True \
    ~trainer.callbacks.save_checkpoint \
    evaluator.config.metrics=[MAE,RMSE] \
    hydra/sweeper=optuna           \
    +optuna_objectives=[MAE,RMSE]   \
    hydra.sweeper.direction=[minimize,minimize] \
    hydra.sweeper.n_trials=16      \
    hydra.sweeper.storage="sqlite:///${RESULTS}/hp_search_multiobjective.db" \
    hydra.sweeper.study_name="${MODEL}_${DATASET}_DIST_${DISTRIBUTED}" \
    ${LAUNCHER}
