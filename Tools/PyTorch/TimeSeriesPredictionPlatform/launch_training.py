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

import warnings
import os
import hydra
from omegaconf import OmegaConf
import torch

import conf.conf_utils
from distributed_utils import is_main_process, init_distributed, init_parallel
from training.utils import set_seed, get_optimization_objectives
from loggers.log_helper import log_parameters
warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="train_config")
def main(config):
    trainer_type = config.trainer._target_

    set_seed(config.get("seed", None))
    model = hydra.utils.instantiate(config.model)

    train, valid, test = hydra.utils.call(config.dataset)
    evaluator = hydra.utils.instantiate(config.evaluator, test_data=test)

    if 'CTLTrainer' in trainer_type:

        init_parallel()
        init_distributed()
        model = model.to(device=config.model.config.device)
        trainer = hydra.utils.instantiate(
            config.trainer,
            optimizer={'params': model.parameters()},
            model=model,
            train_dataset=train,
            valid_dataset=valid,
        )
        log_parameters(trainer.logger, config)

        trainer.train()
        if is_main_process():
            checkpoint = torch.load("best_checkpoint.zip", map_location=evaluator.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            preds, labels, ids, weights = evaluator.predict(model)
            eval_metrics = evaluator.evaluate(preds, labels, ids, weights)
            trainer.logger.log(step=[], data=eval_metrics, verbosity=0)
            trainer.logger.flush()

            del train, valid, test, model, trainer
            torch.cuda.empty_cache()
            objectives = get_optimization_objectives(config, eval_metrics)
            return objectives
    elif 'XGBTrainer' in trainer_type or "StatTrainer" in trainer_type:
        del config.trainer.criterion

        trainer = hydra.utils.instantiate(
            config.trainer,
            model=model,
            train_dataset=train,
            valid_dataset=valid,
        )

        trainer.train()

        preds, labels, ids, weights = evaluator.predict(model)
        eval_metrics = evaluator.evaluate(preds, labels, ids, weights)
        trainer.logger.log(step=[], data=eval_metrics, verbosity=0)
        objectives = get_optimization_objectives(config, eval_metrics)
        return objectives
    else:
        raise AttributeError(f"Not supported Trainer provided {trainer_type}")


if __name__ == "__main__":
    main()
