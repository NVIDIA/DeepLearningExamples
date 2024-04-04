# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

# SPDX-License-Identifier: Apache-2.0
import sys
import gc
import warnings

from hydra.core.hydra_config import HydraConfig
import conf.conf_utils  # loads resolvers
import hydra
import torch
from distributed_utils import is_main_process, init_distributed, init_parallel
from evaluators.evaluator import unpack_predictions
from loggers.log_helper import log_parameters
from training.utils import set_seed, get_optimization_objectives

warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="train_config")
def main(config):
    trainer_type = config.trainer._target_

    set_seed(config.get("seed", None))
    model = hydra.utils.instantiate(config.model)

    train, valid, test = hydra.utils.call(config.dataset)
    evaluator = hydra.utils.instantiate(config.evaluator, test_data=test)
    logger = hydra.utils.call(config.logger)
    log_parameters(logger, config)

    if 'CTLTrainer' in trainer_type:
        init_parallel()
        init_distributed()
        model = model.to(device=config.model.config.device)  # This has to be done before recursive trainer instantiation
        trainer = hydra.utils.instantiate(
            config.trainer,
            optimizer={'params': model.parameters()},
            model=model,
            train_dataset=train,
            valid_dataset=valid,
            logger=logger,
        )

        try:
            trainer.train()
        except RuntimeError as e:
            if 'CUDNN_STATUS_NOT_INITIALIZED' in str(e):
                print(str(e), file=sys.stderr)
                print('This happens sometimes. IDK why. Sorry... Exiting gracefully...', file=sys.stderr)
                logger.log(step=[], data={}, verbosity=0)  # close loggers
                return
            elif 'CUDA out of memory' in str(e):
                print('Job {} caused OOM'.format(HydraConfig.get().job.num), file=sys.stderr)
                print(str(e), file=sys.stderr)
                print('Exiting gracefully...', file=sys.stderr)
                logger.log(step=[], data={}, verbosity=0)  # close loggers
                return
            raise e

        if is_main_process():
            checkpoint = torch.load("best_checkpoint.zip", map_location=evaluator.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            predictions_dict = evaluator.predict(model)
            preds, labels, ids, weights, timestamps, figures = unpack_predictions(predictions_dict)
            eval_metrics = evaluator.evaluate(preds, labels, ids, weights, timestamps)
            logger.log_figures(figures=figures)
            logger.log(step=[], data=eval_metrics, verbosity=0)
            logger.flush()

            # This frees memory when using parallel trainings with joblib. We should stress test it
            # It leaves some memory though which is hard to tell what allocated it. 
            # gc.get_objects() indicate that no tensors are left to collect.
            # joblib's loky backend reuses processes for efficiency reason and prevents PyTorch to cleanup after itself.
            del train, valid, test, model, trainer, evaluator, preds, labels, ids, weights
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

            objectives = get_optimization_objectives(config, eval_metrics)
            return objectives
    elif 'XGBTrainer' in trainer_type:
        del config.trainer.criterion

        trainer = hydra.utils.instantiate(
            config.trainer,
            model=model,
            train_dataset=train,
            valid_dataset=valid,
            logger=logger,
        )

        trainer.train()

        predictions_dict = evaluator.predict(model)
        preds, labels, ids, weights, timestamps, _ = unpack_predictions(predictions_dict)
        eval_metrics = evaluator.evaluate(preds, labels, ids, weights, timestamps)
        logger.log(step=[], data=eval_metrics, verbosity=0)
        objectives = get_optimization_objectives(config, eval_metrics)
        return objectives
    elif "StatTrainer" in trainer_type:
        del config.trainer.criterion

        test.test = True
        trainer = hydra.utils.instantiate(
            config.trainer,
            model=model,
            train_dataset=train,
            valid_dataset=test,
            logger=logger,
            evaluator=evaluator
        )

        predictions_dict = trainer.train()
        preds, labels, ids, weights, timestamps, _ = unpack_predictions(predictions_dict)
        eval_metrics = evaluator.evaluate(preds, labels, ids, weights, timestamps)
        logger.log(step=[], data=eval_metrics, verbosity=0)
        objectives = get_optimization_objectives(config, eval_metrics)
        return objectives
    else:
        raise AttributeError(f"Not supported Trainer provided {trainer_type}")


if __name__ == "__main__":
    main()
