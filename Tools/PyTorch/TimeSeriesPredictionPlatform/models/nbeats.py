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

import numpy as np
import torch
from torch import nn


class Block(nn.Module):

    def __init__(self, units, thetas_dim, backcast_length, forecast_length):
        super(Block, self).__init__()
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        ff = [nn.Linear(backcast_length, units), nn.ReLU()] + [item for _ in range(3) for item in (nn.Linear(units, units), nn.ReLU())]
        self.ff = nn.Sequential(*ff)

        if self.thetas_dim: # generic block skips this stage
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.ff.add_module(str(len(self.ff)), self.theta_b_fc)


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length, forecast_length):

        if not thetas_dim:
            # Auto determine according to paper: horizon/2 sines, horizon/2 cosines
            thetas_dim = forecast_length

        super(SeasonalityBlock, self).__init__(units, thetas_dim, backcast_length,
                                               forecast_length)

        def get_seasonality_basis(num_thetas, linspace):
            p = num_thetas
            p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
            s1 = [np.cos(2 * np.pi * i * linspace) for i in range(p1)]
            s2 = [np.sin(2 * np.pi * i * linspace) for i in range(p2)]
            s = np.stack(s1+s2)
            return torch.FloatTensor(s)

        self.forecast_length = forecast_length
        linspace = np.concatenate([np.arange(backcast_length) / backcast_length, np.arange(forecast_length) / forecast_length])
        self.register_buffer('basis', get_seasonality_basis(self.thetas_dim, linspace))

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = self.ff(x)
        x = x.mm(self.basis)
        backcast, forecast = x[:,:-self.forecast_length], x[:,-self.forecast_length:]
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length, forecast_length):
        super(TrendBlock, self).__init__(units, thetas_dim, backcast_length,
                                         forecast_length)

        self.forecast_length = forecast_length
        linspace = np.concatenate([np.arange(backcast_length) / backcast_length, np.arange(forecast_length) / forecast_length])
        basis = np.stack([linspace ** i for i in range(thetas_dim)])
        self.register_buffer('basis', torch.FloatTensor(basis))

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = self.ff(x)
        x = x.mm(self.basis)
        backcast, forecast = x[:, :-self.forecast_length], x[:, -self.forecast_length:]
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length, forecast_length):

        super(GenericBlock, self).__init__(units, None, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(units, backcast_length)
        self.forecast_fc = nn.Linear(units, forecast_length)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = self.ff(x)

        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)

        return backcast, forecast

class NBeatsNet(nn.Module):
    BLOCK_MAP = {'seasonality': SeasonalityBlock,
                 'trend': TrendBlock,
                 'generic': GenericBlock
                 }

    def __init__(self, config):
        super(NBeatsNet, self).__init__()
        model_config = config

        self.forecast_length = config.example_length - config.encoder_length
        self.backcast_length = config.encoder_length
        self.stacks = nn.ModuleList([self.create_stack(c) for c in config.stacks])

    def create_stack(self, stack_config):
        blocks = nn.ModuleList()

        for block_id in range(stack_config.num_blocks):
            block_init = NBeatsNet.BLOCK_MAP[stack_config.type]
            if stack_config.share_weights and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(units = stack_config.hidden_size,
                                   thetas_dim=stack_config.theta_dim,
                                   backcast_length=self.backcast_length,
                                   forecast_length=self.forecast_length)
            blocks.append(block)
        return blocks

    def forward(self, batch_dict):
        backcast = batch_dict['target'][:,:self.backcast_length,:]
        backcast = squeeze_last_dim(backcast)
        forecast = backcast.new_zeros(size=(backcast.size()[0], self.forecast_length,))
        for stack in self.stacks:
            for block in stack:
                b, f = block(backcast)
                backcast = backcast - b
                forecast = forecast + f
        forecast = forecast.unsqueeze(2)
        return forecast


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def linear_space(backcast_length, forecast_length):
    ls = np.arange(-backcast_length, forecast_length, 1) / forecast_length
    b_ls = ls[:backcast_length]
    f_ls = ls[backcast_length:]
    return b_ls, f_ls
