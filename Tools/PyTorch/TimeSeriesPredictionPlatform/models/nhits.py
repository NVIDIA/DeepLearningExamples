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

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ['linear','nearest']) or ('cubic' in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = theta[:, :self.backcast_size]
        knots = theta[:, self.backcast_size:]

        if self.interpolation_mode in ['nearest','linear']:
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:,0,:]
        elif 'cubic' in self.interpolation_mode:
            batch_size = len(backcast)
            knots = knots[:,None,None,:]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots)/batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i*batch_size:(i+1)*batch_size], size=self.forecast_size, mode='bicubic')
                forecast[i*batch_size:(i+1)*batch_size] += forecast_i[:,0,0,:]

        return backcast, forecast

ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']

POOLING = ['MaxPool1d',
           'AvgPool1d']

class NHITSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """
    def __init__(self, 
                 input_size: int,
                 n_theta: int, 
                 n_mlp_layers: int,
                 hidden_size: int,
                 basis: nn.Module,
                 n_pool_kernel_size: int,
                 pooling_mode: str,
                 dropout_prob: float,
                 activation: str):
        """
        """
        super().__init__()

        n_time_in_pooled = int(np.ceil(input_size/n_pool_kernel_size))

        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        assert pooling_mode in POOLING, f'{pooling_mode} is not in {POOLING}'

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(kernel_size=n_pool_kernel_size,
                                                       stride=n_pool_kernel_size, ceil_mode=True)

        # Block MLPs
        mlp = [nn.Linear(n_time_in_pooled, hidden_size)]
        mlp += [item for _ in range(n_mlp_layers) for item in (nn.Linear(hidden_size, hidden_size), activ)]
        layers = mlp + [nn.Linear(hidden_size, n_theta)]

        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Pooling
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast

class NHITS(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert len(config.n_pool_kernel_size) == len(config.n_freq_downsample) == len(config.n_blocks)

        self.config = config
        self.input_size = config.encoder_length
        self.h = config.example_length - config.encoder_length

        blocks = self.create_stack(config)
        self.blocks = torch.nn.ModuleList(blocks)

    def create_stack(self, config):                     

        block_list = []
        for n, k, f in zip(config.n_blocks, config.n_pool_kernel_size, config.n_freq_downsample):
            for _ in range(n):
                n_theta = (self.input_size + max(self.h//f, 1) )
                basis = _IdentityBasis(backcast_size=self.input_size,
                                       forecast_size=self.h,
                                       interpolation_mode=config.interpolation_mode)                 
                nbeats_block = NHITSBlock(input_size=self.input_size,
                                          n_theta=n_theta,
                                          n_mlp_layers=config.n_mlp_layers,
                                          hidden_size=config.hidden_size,
                                          n_pool_kernel_size=k,
                                          pooling_mode=config.pooling_mode,
                                          basis=basis,
                                          dropout_prob=config.dropout_prob_theta,
                                          activation=config.activation)
                block_list.append(nbeats_block)

        return block_list

    def forward(self, batch):
        residuals = batch['target'][:, :self.input_size, 0]

        forecast = residuals[:, -1:] # Level with Naive1
        block_forecasts = [ forecast.repeat(1, self.h) ]

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        return forecast.unsqueeze(2)
