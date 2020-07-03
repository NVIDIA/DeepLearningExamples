# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Sequence, List, Iterable

import apex.mlp
import torch
from torch import nn


class AbstractMlp(nn.Module):
    """
    MLP interface used for configuration-agnostic checkpointing (`dlrm.utils.checkpointing`)
    and easily swappable MLP implementation
    """

    @property
    def weights(self) -> List[torch.Tensor]:
        """
        Getter for all MLP layers weights (without biases)
        """
        raise NotImplementedError()

    @property
    def biases(self) -> List[torch.Tensor]:
        """
        Getter for all MLP layers biases
        """
        raise NotImplementedError()

    def forward(self, mlp_input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def load_state(self, weights: Iterable[torch.Tensor], biases: Iterable[torch.Tensor]):
        for new_weight, weight, new_bias, bias in zip(weights, self.weights, biases, self.biases):
            weight.data = new_weight.data
            weight.data.requires_grad_()

            bias.data = new_bias.data
            bias.data.requires_grad_()


class TorchMlp(AbstractMlp):
    def __init__(self, input_dim: int, sizes: Sequence[int]):
        super().__init__()

        layers = []
        for output_dims in sizes:
            layers.append(nn.Linear(input_dim, output_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dim = output_dims

        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                nn.init.normal_(module.bias.data, 0., math.sqrt(1. / module.out_features))

    @property
    def weights(self):
        return [layer.weight for layer in self.layers if isinstance(layer, nn.Linear)]

    @property
    def biases(self):
        return [layer.bias for layer in self.layers if isinstance(layer, nn.Linear)]

    def forward(self, mlp_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mlp_input (Tensor): with shape [batch_size, num_features]

        Returns:
            Tensor: Mlp output in shape [batch_size, num_output_features]
        """
        return self.layers(mlp_input)


class CppMlp(AbstractMlp):

    def __init__(self, input_dim: int, sizes: Sequence[int]):
        super().__init__()

        self.mlp = apex.mlp.MLP([input_dim] + list(sizes))

    @property
    def weights(self):
        return self.mlp.weights

    @property
    def biases(self):
        return self.mlp.biases

    def forward(self, mlp_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mlp_input (Tensor): with shape [batch_size, num_features]

        Returns:
            Tensor: Mlp output in shape [batch_size, num_output_features]
        """
        return self.mlp(mlp_input)
