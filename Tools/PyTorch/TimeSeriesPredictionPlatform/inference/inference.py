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

import logging
import os
from typing import Dict, List, Optional, Tuple

import dllogger
import hydra
import numpy as np
import torch
import importlib
try:
    from apex import amp
except ImportError:
    print("Nvidia apex not available. Can't use apex Automatic Mixed Precision (AMP) for training.\
    Please check: https://github.com/NVIDIA/apex for installation")
from omegaconf import OmegaConf

import conf.conf_utils
from data.data_utils import Preprocessor
from evaluators.evaluator import unpack_predictions


def run_inference(config):
    cfg = config
    with open(os.path.join(cfg.checkpoint, ".hydra/config.yaml"), "rb") as f:
        config = OmegaConf.load(f)
    if cfg.get("evaluator", None) is not None:
        config.evaluator.config = OmegaConf.merge(config.evaluator.config, cfg.evaluator.config)
    if cfg.get("dataset_dir", None):
        if not os.path.isdir(config.dataset.config.dest_path):
            raise ValueError("dataset_dir must be a directory")
        config.dataset.config.dest_path = cfg.dataset_dir
    config.evaluator.config.device = cfg.device
    if cfg.get("dataset_path", None):
        preprocessor = Preprocessor(config.dataset.config)
        if cfg.get("preproc_state_path", None):
            preprocessor_state_file = cfg.preproc_state_path
        else:
            preprocessor_state_file = None
        preprocessor.load_state(preprocessor_state_file)
        test_df = preprocessor.preprocess_test(dataset=cfg.dataset_path)
        test_df = preprocessor.apply_scalers(test_df)
        test_df = preprocessor.impute(test_df)
        train, valid, test = hydra.utils.call(config.dataset, input_df=test_df)
    else:
        train, valid, test = hydra.utils.call(config.dataset)
    del train, valid
    evaluator = hydra.utils.instantiate(config.evaluator, test_data=test)
    model = hydra.utils.instantiate(config.model)
    if not (config.dataset.config.get('xgb', False) or config.dataset.config.get('stat', False)):
        state_dict = torch.load(os.path.join(cfg.checkpoint, "best_checkpoint.zip"))['model_state_dict']
        model.load_state_dict(state_dict)
        device = torch.device(cfg.device)  # maybe change depending on evaluator
        model.to(device=device)
        precision = cfg.precision
        assert precision in ["fp16", "fp32"], "Precision needs to be either fp32 or fp16"
        if precision == "fp16" and importlib.util.find_spec("apex"):
            model = amp.initialize(model, opt_level="O2")
    else:
        model.load(cfg.checkpoint)

    predictions_dict = evaluator.predict(model)
    preds, labels, ids, weights, timestamps, _ = unpack_predictions(predictions_dict)
    eval_metrics = evaluator.evaluate(preds, labels, ids, weights, timestamps)
    logger = hydra.utils.call(config.logger)
    logger.log(step=[], data={k: float(v) for k, v in eval_metrics.items()}, verbosity=dllogger.Verbosity.VERBOSE)
    logger.log(step='event', data={"String": "Evaluation Metrics: {}".format(eval_metrics)}, verbosity=dllogger.Verbosity.DEFAULT)
    return eval_metrics


