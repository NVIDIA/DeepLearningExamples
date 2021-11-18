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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm
from torch import Tensor

from data.data_utils import DataTypes, InputTypes, translate_features
from models.tft_pyt.modeling import *


class LSTM(nn.Module):
    """ 
    Implementation from LSTM portion of https://arxiv.org/abs/1912.09363 
    """

    def __init__(self, config):
        super().__init__()

        self.encoder_steps = config.dataset.encoder_length  # this determines from how distant past we want to use data from

        self.mask_nans = config.model.missing_data_strategy == "mask"
        self.features = translate_features(config.dataset.features)
        config = config.model

        config.temporal_known_continuous_inp_size = len(
            [x for x in self.features if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS]
        )
        config.temporal_observed_continuous_inp_size = len(
            [
                x
                for x in self.features
                if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS
            ]
        )
        config.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        config.static_continuous_inp_size = len(
            [
                x
                for x in self.features
                if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS
            ]
        )
        config.static_categorical_inp_lens = [
            x.get("cardinality", 100)
            for x in self.features
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CATEGORICAL
        ]

        config.temporal_known_categorical_inp_lens = [
            x.get("cardinality", 100)
            for x in self.features
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CATEGORICAL
        ]
        config.temporal_observed_categorical_inp_lens = [
            x.get("cardinality", 100)
            for x in self.features
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CATEGORICAL
        ]

        config.num_static_vars = config.static_continuous_inp_size + len(config.static_categorical_inp_lens)
        config.num_future_vars = config.temporal_known_continuous_inp_size + len(config.temporal_known_categorical_inp_lens)
        config.num_historic_vars = sum(
            [
                config.num_future_vars,
                config.temporal_observed_continuous_inp_size,
                config.temporal_target_size,
                len(config.temporal_observed_categorical_inp_lens),
            ]
        )

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
        future_inputs = t_known_inp[:, self.encoder_steps :]

        # Encoders
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc))
        future_features, _ = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)

        output = self.output_proj(future)
        return output
