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

import torch
import torch.nn as nn
from torch import Tensor

from models.tft_pyt.modeling import *


class LSTM(nn.Module):
    """
    Implementation from LSTM portion of https://arxiv.org/abs/1912.09363
    """

    def __init__(self, config):
        super().__init__()

        self.encoder_steps = config.encoder_length  # this determines from how distant past we want to use data from

        self.mask_nans = config.missing_data_strategy == "mask"

        self.embedding = TFTEmbedding(config)
        self.static_encoder = StaticCovariateEncoder(config)

        self.history_vsn = VariableSelectionNetwork(config, config.num_historic_vars)
        self.history_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.future_vsn = VariableSelectionNetwork(config, config.num_future_vars)
        self.future_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)

        self.output_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        s_inp, t_known_inp, t_observed_inp, t_observed_tgt = self.embedding(x)

        # Static context
        cs, ce, ch, cc = self.static_encoder(s_inp)
        ch, cc = ch.unsqueeze(0), cc.unsqueeze(0)  # lstm initial states

        # Temporal input
        _historical_inputs = [t_known_inp[:, : self.encoder_steps, :], t_observed_tgt[:, : self.encoder_steps, :]]
        if t_observed_inp is not None:
            _historical_inputs.insert(0, t_observed_inp[:, : self.encoder_steps, :])

        historical_inputs = torch.cat(_historical_inputs, dim=-2)
        future_inputs = t_known_inp[:, self.encoder_steps:]

        # Encoders
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc))
        future_features, _ = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)

        output = self.output_proj(future)
        return output
