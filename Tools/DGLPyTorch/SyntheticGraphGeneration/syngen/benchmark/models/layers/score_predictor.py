# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
   Predictor layer
"""


class ScorePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [
            nn.Linear(
                input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True
            )
            for l in range(L)
        ]
        list_FC_layers.append(
            nn.Linear(input_dim // 2 ** L, output_dim, bias=True)
        )
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
