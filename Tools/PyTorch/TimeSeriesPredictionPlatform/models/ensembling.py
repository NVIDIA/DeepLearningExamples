# Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved.
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

import hydra
from omegaconf import OmegaConf
import torch
from torch import nn, Tensor
from typing import Dict
from training.utils import to_device


class ModelEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reduction_stategy = config.get('reduction_strategy', 'mean')
        self.model_weights = []
        self.model_list = []
        for model_info in self.config.model_list:
            self.model_weights.append(model_info.get('weight', 1.0))

            model_dir = model_info.dir
            with open(os.path.join(model_dir, '.hydra/config.yaml'), 'rb') as f:
                cfg = OmegaConf.load(f)
            model: nn.Module = hydra.utils.instantiate(cfg.model)
            if not(cfg.dataset.config.get('xgb', False) or cfg.dataset.config.get('stat', False)):
                # reduce gpu memory usage
                state_dict = torch.load(os.path.join(model_dir, model_info.get('checkpoint', 'best_checkpoint.zip')), map_location='cpu')['model_state_dict']
                model.load_state_dict(state_dict)
            else:
                raise ValueError('XGB and stat models are currently not supported by ensembling.')
            self.model_list.append(model)

        self.num_devices = min(torch.cuda.device_count(), len(self.model_list))
        model_splits = [self.model_list[i::self.num_devices] for i in range(self.num_devices)]
        model_splits = [nn.ModuleList(x).to(f'cuda:{i}') for i, x in enumerate(model_splits)]
        self.model_splits = nn.ModuleList(model_splits)

        self.model_weights = [x for y in [self.model_weights[i::self.num_devices] for i in range(self.num_devices)] for x in y]

    @staticmethod
    def _reduce_preds(preds, weights, reduction_stategy):
        if reduction_stategy not in ['mean', 'sum']:
            raise ValueError(f'Unknown reduction strategy: {reduction_stategy}')

        result = sum(p * w for p, w in zip(preds, weights))
        if reduction_stategy == 'mean':
            result /= sum(weights)

        return result

    @staticmethod
    def _replicate(batch, n):
        return [to_device(batch, i) for i in range(n)]

    @torch.no_grad()
    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        _x = self._replicate(x, self.num_devices)
        preds = [[] for _ in range(self.num_devices)]

        for i, (data, split) in enumerate(zip(_x, self.model_splits)):
            for model in split:
                test_method_name = 'predict' if hasattr(model, 'predict') else '__call__'
                test_method = getattr(model, test_method_name)
                pred = test_method(data)
                # Move all preds to cpu. Probably it will have a terrible performance, but we have tons of memory there
                preds[i].append(pred.cpu())

        preds = [x for y in preds for x in y]
        preds = self._reduce_preds(preds, self.model_weights, self.reduction_stategy)

        return preds

class XGBEnsemble:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reduction_stategy = config.get('reduction_strategy', 'mean')
        self.model_weights = []
        self.model_list = []
        for model_info in self.config.model_list:
            self.model_weights.append(model_info.get('weight', 1.0))

            model_dir = model_info.dir
            with open(os.path.join(model_dir, '.hydra/config.yaml'), 'rb') as f:
                cfg = OmegaConf.load(f)
            model: nn.Module = hydra.utils.instantiate(cfg.model)
            model.load(model_dir)
            self.model_list.append(model)

    @staticmethod
    def _reduce_preds(preds, weights, reduction_stategy):
        if reduction_stategy not in ['mean', 'sum']:
            raise ValueError(f'Unknown reduction strategy: {reduction_stategy}')

        result = sum(p * w for p, w in zip(preds, weights))
        if reduction_stategy == 'mean':
            result /= sum(weights)

        return result

    def predict(self, x, i):
        preds = []

        for model in self.model_list:
            pred = model.predict(x, i)
            preds.append(pred)

        preds = self._reduce_preds(preds, self.model_weights, self.reduction_stategy)

        return preds
