# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from functools import partial
from typing import Dict, List, Optional, Tuple

import dgl
import dllogger
import hydra
import numpy as np
import torch
from apex import amp
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from conf.conf_utils import append_derived_config_fields
from loggers.log_helper import setup_logger
from training.utils import to_device


def run_inference(config):
    cfg = config
    state_dict = torch.load(os.path.join(cfg.evaluator.checkpoint, "best_checkpoint.pth.tar"))["model_state_dict"]
    device = torch.device(cfg.device.get("name", "cpu"))  # maybe change depending on evaluator
    with open(os.path.join(cfg.evaluator.checkpoint, ".hydra/config.yaml"), "rb") as f:
        config = OmegaConf.load(f)
        append_derived_config_fields(config)

    if config.config.device.get("world_size", 1) > 1:
        model_params = list(state_dict.items())
        for k, v in model_params:
            if k[:7] == "module.":
                state_dict[k[7:]] = v
                del state_dict[k]
    config.config.evaluator = OmegaConf.merge(config.config.evaluator, cfg.evaluator)
    if cfg.inference.get("dataset_dir", None):
        config.config.dataset.dest_path = cfg.inference.dataset_dir
    config._target_ = config.config.evaluator._target_
    evaluator = hydra.utils.instantiate(config)
    config._target_ = config.config.model._target_
    config.config.device = cfg.device
    model = hydra.utils.instantiate(config)
    test_method_name = config.config.model.get("test_method", "__call__")
    test_method = getattr(model, test_method_name)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device=device)
    precision = cfg.inference.precision
    assert precision in ["fp16", "fp32"], "Precision needs to be either fp32 or fp16"
    if precision == "fp16":
        model = amp.initialize(model, opt_level="O2")
    if os.path.isdir(config.config.dataset.dest_path):
        config._target_ = config.config.dataset._target_
        train, valid, test = hydra.utils.call(config)
        del train
        del valid
    else:
        raise ValueError("dataset_dir must be a directory")
    preds_full = []
    labels_full = []
    weights_full = []
    ids_full = []
    test_target = "target_masked" if config.config.model.get("test_target_mask", True) else "target"
    preds_test_output_selector = config.config.model.get("preds_test_output_selector", -1)
    if config.config.dataset.get("graph", False) and config.config.model.get("graph_eligible", False):

        def _collate_graph(samples, target):
            batch = dgl.batch(samples)
            labels = batch.ndata["target"]
            # XXX: we need discuss how to do this neatly
            if target == "target_masked":
                labels = labels[:, config.config.dataset.encoder_length :, :]

            return batch, labels

        _collate = _collate_graph
    else:

        def _collate_dict(samples, target):
            batch = default_collate(samples)
            labels = batch["target"]
            if target == "target_masked":
                labels = labels[:, config.config.dataset.encoder_length :, :]
            return batch, labels

        _collate = _collate_dict
    data_loader = DataLoader(
        test,
        batch_size=int(cfg.inference.batch_size),
        num_workers=2,
        pin_memory=True,
        collate_fn=partial(_collate, target=test_target),
    )
    with torch.no_grad():
        for i, (batch, labels) in enumerate(data_loader):

            batch = to_device(batch, device=device)
            labels = to_device(labels, device=device)

            if cfg.evaluator.get("use_weights", False):
                weights = batch["weight"]
            else:
                weights = None
            ids = batch["id"]

            labels_full.append(labels)
            weights_full.append(weights)
            preds = test_method(batch)
            if preds_test_output_selector >= 0:
                preds = preds[..., preds_test_output_selector : preds_test_output_selector + 1]
            ids_full.append(ids)
            preds_full.append(preds)

        preds_full = torch.cat(preds_full, dim=0).cpu().numpy()
        labels_full = torch.cat(labels_full, dim=0).cpu().numpy()
        if cfg.evaluator.get("use_weights", False):
            weights_full = torch.cat(weights_full).cpu().numpy()
        else:
            weights_full = np.zeros((0, 0))
        ids_full = torch.cat(ids_full).cpu().numpy()
        eval_metrics = evaluator(labels_full, preds_full, weights_full, ids_full)
    logger = setup_logger(cfg)
    logger.log(step=[], data={k: float(v) for k, v in eval_metrics.items()}, verbosity=dllogger.Verbosity.VERBOSE)
    logger.log(step=[], data={"String": "Evaluation Metrics: {}".format(eval_metrics)}, verbosity=dllogger.Verbosity.DEFAULT)
    return eval_metrics
