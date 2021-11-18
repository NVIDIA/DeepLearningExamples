# SPDX-License-Identifier: Apache-2.0

import logging
import warnings

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from conf.conf_utils import append_derived_config_fields
from data.data_utils import StatDataset
from distributed_utils import is_main_process
from training.trainer import CTLTrainer, StatTrainer

warnings.filterwarnings("ignore")


def set_seed(seed):
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    append_derived_config_fields(cfg)
    set_seed(cfg.config.trainer.get("seed", None))
    if cfg.config.get("save_config", False):
        with open(cfg.config.get("save_path", "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f)
            return
    if cfg.config.trainer.get("type", "") != "stat":
        device = torch.device(cfg.config.device.get("name", "cpu"))
        cfg._target_ = cfg.config.dataset._target_
        train, valid, test = hydra.utils.call(cfg)
        cfg._target_ = cfg.config.model._target_
        model = hydra.utils.instantiate(cfg)
        cfg._target_ = cfg.config.optimizer._target_
        optimizer = hydra.utils.instantiate(cfg, params=model.parameters())
        cfg._target_ = cfg.config.criterion._target_
        criterion = hydra.utils.call(cfg)
        cfg._target_ = cfg.config.evaluator._target_
        evaluator = hydra.utils.instantiate(cfg)
        trainer = CTLTrainer(model, train, valid, test, optimizer, evaluator, criterion, cfg.config)
        trainer.train()
        if is_main_process():
            eval_metrics = trainer.evaluate()
        torch.cuda.synchronize()
        del train, valid, test
    else:
        dataset = StatDataset(
            cfg.config.dataset.features,
            csv_path=cfg.config.dataset.dest_path,
            encoder_length=cfg.config.dataset.encoder_length,
            example_length=cfg.config.dataset.example_length,
            stride=cfg.config.dataset.get("stride", 1),
            split=cfg.config.dataset.test_range[0],
            split_feature=cfg.config.dataset.time_ids,
        )
        cfg._target_ = cfg.config.model._target_
        model = hydra.utils.instantiate(cfg)
        cfg._target_ = cfg.config.evaluator._target_
        evaluator = hydra.utils.instantiate(cfg)
        trainer = StatTrainer(dataset, evaluator, cfg.config, model)
        eval_metrics = trainer.evaluate()
        logging.info(eval_metrics)


if __name__ == "__main__":
    main()
