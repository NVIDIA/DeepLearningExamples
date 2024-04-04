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
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPP_criterion_wrapper(nn.Module):
    '''This wrapper unifies definition of forward function across different criterions'''
    def __init__(self, criterion, cl_start_horizon=None, cl_update=None):
        super().__init__()
        self.criterion = criterion
        self.base_arguments = set(criterion.forward.__code__.co_varnames)
        self.additional_arguments = {'weights'}
        self.allowed_arguments = self.base_arguments.union(self.additional_arguments)

        # Curriciulum learning
        assert bool(cl_start_horizon) == bool(cl_update), "Both cl_start_horizon and cl_update have to be set or unset at the same time"
        self.curr_horizon = cl_start_horizon
        self.horizon_update = cl_update
        self.cl_counter = 0

    def forward(self, preds, labels, weights=None, **kwargs):
        """
        preds: Tensor of size BS x time x num_targets x num_estimators
               or BS x time x num_ids x num_targets x num_estimators in case of MultiTarget dataset
        labels: Tensor of size BS x time x num_targets
               or BS x time x num_ids x num_targets case of MultiTarget dataset
        weights: Tensor of the same shape as labels
        """
        disallowed_kwargs = set(kwargs.keys()) - self.allowed_arguments
        if disallowed_kwargs:
            raise TypeError(f'Invalid keyword arguments {disallowed_kwargs} for {type(self.criterion)}')
        kwargs = {name:arg for name, arg in kwargs.items() if name in self.base_arguments}

        if self.training:
            if self.curr_horizon:
                preds = preds[:, :self.curr_horizon]
                labels = labels[:, :self.curr_horizon]
                weights = weights[:, :self.curr_horizon] if weights is not None else None
                if (self.cl_counter + 1) % self.horizon_update == 0:
                    self.curr_horizon += 1
                self.cl_counter += 1
    
        if len(preds.shape) == 4 and len(labels.shape) == 3:
            labels = labels.unsqueeze(-1)
            if weights is not None:
                weights = weights.unsqueeze(-1)

        loss = self.criterion(preds, labels, **kwargs) 
        if weights is not None and weights.numel():
            loss *= weights
            loss = loss.view(-1, *loss.shape[2:]).mean(0)

        return loss


class QuantileLoss(nn.Module):
    def __init__(self, quantiles, reduction='mean'):
        super().__init__()
        self.quantiles = quantiles
        self.reduce = reduction == 'mean'

    def forward(self, predictions, targets):
        if not hasattr(self, 'q'):
            self.register_buffer('q', predictions.new(self.quantiles))
        diff = predictions - targets
        losses = (1-self.q) * F.relu(diff) + self.q * F.relu(-diff)
        if self.reduce:
            losses = losses.view(-1, losses.shape[-1]).mean(0)
        return losses
    
class GaussianLogLikelihood(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduce = reduction == 'mean'

    def forward(self, predictions, targets):
        # Inputs with shape  [BS, window, 2] (mean + std)
        # Targets with shape [BS, window, 1]

        mu = predictions[..., 0:1]
        sig = predictions[..., 1:2]
        var = sig ** 2
        loss =  -((targets - mu) ** 2) / (2 * var) - sig.log()
        if self.reduce:
            loss = loss.mean(0)
        return -loss


class TweedieLoss(nn.Module):
    def __init__(self, reduction='mean', p=1.1):
        super().__init__()
        assert 1.0 < p < 2.0, 'Variance power should be in 1..2 interval'
        self.reduce = reduction == 'mean'
        self.register_buffer('variance_power', torch.tensor(p))
    
    def forward(self, predictions, targets):
        # Inputs with shape [BS, window, 1]
        # Targets with shape [BS, window, 1]

        rho = self.get_buffer('variance_power').to(device=predictions.device)
        predictions[predictions < 1e-10] = 1e-10
        log_preds = torch.log(predictions) 
        likelihood = -targets * torch.exp(log_preds * (1 - rho)) / (1 - rho) + torch.exp(log_preds * (2 - rho)) / (2 - rho)

        likelihood = likelihood.view(targets.shape[0], targets.shape[1])

        loss = torch.unsqueeze(likelihood, -1)

        if self.reduce:
            loss = loss.mean(0)
        return loss

