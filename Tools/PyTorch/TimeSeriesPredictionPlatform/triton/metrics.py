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
        
        train, valid, test = hydra.utils.call(self.config.dataset)
        del train, valid
        self.evaluator = hydra.utils.call(self.config.evaluator, test_data=test)
        self.predictions = []
        self.targets = []
        self.ids = []
        self.weights = []


    @property
    def metrics(self):
        targets = np.concatenate(self.targets, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        weights = np.concatenate(self.weights, axis=0)
        ids = np.concatenate(self.ids, axis=0)
        if np.isnan(weights).any():
            weights = np.empty([0])
        return self.evaluator.evaluate(targets, predictions, ids, weights)

    def update(
        self,
        ids,
        y_pred,
        x,
        y_real,
    ):
        self.targets.append(y_real['target__0'][:,:,0][:,:,np.newaxis])
        self.ids.append(ids)
        self.weights.append(x["weight__9"])
        preds = y_pred["target__0"]
        self.predictions.append(preds)
