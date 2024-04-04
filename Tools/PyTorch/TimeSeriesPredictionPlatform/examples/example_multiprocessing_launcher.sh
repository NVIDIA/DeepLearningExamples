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

# More info here: https://hydra.cc/docs/plugins/optuna_sweeper/
: ${DATASET:=electricity}
RESULTS=/ws/tft_${DATASET}_hp_search
mkdir -p ${RESULTS}

python launch_training.py \
    -m \
    'model.config.n_head=choice(1,2)' \
    'model.config.hidden_size=choice(96,128)' \
    'model.config.dropout=interval(0,0.5)' \
    'trainer.optimizer.lr=tag(log, interval(1e-5, 1e-2))' \
    'trainer.config.ema=choice(true, false)' \
    '+trainer.config.ema_decay=interval(0.9, 0.9999)' \
    model=tft \
    dataset=${DATASET}            \
    trainer/criterion=quantile     \
    trainer.config.batch_size=1024 \
    trainer.config.num_epochs=10    \
    trainer.config.log_interval=-1 \
    trainer.config.mlflow_store="file://${RESULTS}/mlruns" \
    evaluator.config.metrics=[MAE,RMSE,SMAPE,TDI] \
    hydra/sweeper=optuna           \
    +optuna_objectives=[MAE,RMSE,SMAPE,TDI]   \
    hydra.sweeper.direction=[minimize,minimize,minimize,minimize] \
    hydra.sweeper.n_trials=8      \
    hydra.sweeper.experiment_sequence=hydra_utils.TSPPOptunaExperimentSequence \
    hydra.launcher.n_jobs=4 \
    hydra.sweeper.storage="sqlite:///${RESULTS}/hp_search_multiobjective.db" \
    hydra.sweeper.study_name="tft_${DATASET}_1GPU" \
    hydra/launcher=multiprocessing \
    hydra.sweep.dir='/results/${now:%Y-%m-%d}/${now:%H-%M-%S}'
