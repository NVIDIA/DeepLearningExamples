# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import os
import pandas as pd
import numpy as np
import pickle
import argparse
import hydra
import torch
from triton.deployment_toolkit.core import BaseMetricsCalculator
from omegaconf import OmegaConf

def update_argparser(parser):
    parser.add_argument("--model-dir", type=str, help="Path to the model directory you would like to use (likely in outputs)", required=True)





class MetricsCalculator(BaseMetricsCalculator):
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, ".hydra/config_merged.yaml"), "rb") as f:
            self.config = OmegaConf.load(f)
        self.config._target_ = self.config.config.evaluator._target_
        self.evaluator = hydra.utils.call(self.config)
        self.config= self.config.config
        self.output_selector = self.config.model.get("preds_test_output_selector", -1)
        self.predictions = []
        self.targets = []
        self.ids = []
        if self.config.evaluator.get("use_weights", False):
            self.weights = []


    @property
    def metrics(self):
        targets = np.concatenate(self.targets, axis=0)
        # targets = torch.cat(self.targets, dim=0)
        predictions = np.concatenate(self.predictions, axis=0)
        # predictions = torch.cat(self.predictions, dim=0)

        ids = np.concatenate(self.ids, axis=0)
        if self.config.evaluator.get("use_weights", False):
            weights = torch.cat(self.weights).cpu().numpy()
        else:
            weights = np.zeros((0, 0))
        return self.evaluator(targets, predictions, weights, ids=ids)

    def update(
        self,
        ids,
        y_pred,
        x,
        y_real,
    ):
        #can probably just pass all of this to the evaluator main class
        self.targets.append(y_real['target__0'][:,:,0][:,:,np.newaxis])
        self.ids.append(ids)
        if self.config.evaluator.get("use_weights", False):
            self.weights.append(x["weight"])
        preds = y_pred["target__0"]
        self.predictions.append(preds)

        # return self.metrics
