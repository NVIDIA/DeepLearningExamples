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

'''Defines the neural network, loss function and metrics'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .tft_pyt.modeling import LazyEmbedding
import torch
from torch import nn

class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, num_layers, dropout, tgt_embed):
        super(AutoregressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.tgt_embed = tgt_embed

        # This is a modification to the more general algorithm implemented here that uses all the layer's hidden states to make a final prediction
        # This is not what is described in the paper but is what reference implementation did
        # In this particular case it is used for expected value (mu) estimation
        self.mu_proj = nn.Linear(hidden_size * num_layers, 1)
        self.sig_proj = nn.Sequential(
            nn.Linear(hidden_size * num_layers, 1),
            nn.Softplus()
        )

        
    def forward(self, inputs, embedded_labels, hidden=None, mask=None):
        # Inputs should be all covariate embeddings and embedded labels should be target emdeddings

        mus = []
        sigs = []
        for i in range(inputs.shape[1]):
            input = inputs[:,i]
            if embedded_labels is None or mask is None:
                mu_embed = self.tgt_embed(mu)
                input = torch.cat((input, mu_embed), dim=-1)
            elif i and mask[:,i].any():
                mu_embed = self.tgt_embed(mu)
                input = torch.cat((input, torch.where(mask[:, i], mu_embed, embedded_labels[:, i])), dim=-1)
            else:
                input = torch.cat((input, embedded_labels[:, i]), dim=-1)

            _, hidden = self.lstm(input.unsqueeze(1), hidden)
            hidden_permute = hidden[0].permute(1, 2, 0).contiguous().view(hidden[0].shape[1], -1)
            mu = self.mu_proj(hidden_permute)

            sig = self.sig_proj(hidden_permute)

            mus.append(mu)
            sigs.append(sig)

        mus = torch.cat(mus, dim=1)
        sigs = torch.cat(sigs, dim=1)
        return mus, sigs, hidden


class DeepAR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder_length = config.encoder_length
        self.example_length = config.example_length
        self.register_buffer('quantiles', torch.FloatTensor(config.quantiles), persistent=False)
        self.use_embedding = self.config.use_embedding
        self.drop_variance = self.config.get('drop_variance', False)

        _config = config.copy()
        _config.hidden_size = self.config.embedding_dim
        _config.num_historic_vars -= len(config.temporal_observed_categorical_inp_lens)
        _config.num_historic_vars -= config.temporal_observed_continuous_inp_size
        _config.temporal_observed_categorical_inp_lens = []
        _config.temporal_observed_continuous_inp_size = 0

        self.embedding_v2 = LazyEmbedding(_config)
        tgt_embed = lambda x: torch.matmul(x, self.embedding_v2.t_tgt_embedding_vectors) + self.embedding_v2.t_tgt_embedding_bias

        inp_size = (config.num_static_vars + config.num_future_vars + config.temporal_target_size) * config.embedding_dim

        self.encoder = AutoregressiveLSTM(input_size=inp_size,
                                            hidden_size=config.hidden_size, 
                                            embed_size=config.embedding_dim, 
                                            num_layers=config.num_layers,
                                            dropout=config.dropout,
                                            tgt_embed=tgt_embed)

    def _roll_data(x):
        if x is None:
            return None
        x = torch.roll(x, 1, 1)
        x[:,0] = 0
        return x


    def forward(self, batch, predict=False):
        batch = batch.copy() # shallow copy to replace observables in this scope
        batch['target'] = DeepAR._roll_data(batch['target'])
        batch['weight'] = DeepAR._roll_data(batch['weight'])
        batch['o_cat'] = None
        batch['o_cont'] = None

        s_emb, k_emb, _, tgt_emb = self.embedding_v2(batch)
        s_emb = s_emb.unsqueeze(1).expand(s_emb.shape[0], tgt_emb.shape[1], *s_emb.shape[1:])

        feat  = torch.cat((s_emb, k_emb) , axis=-2)
        feat = feat.view(*feat.shape[:-2], -1)
        tgt_emb = tgt_emb.view(*tgt_emb.shape[:-2], -1)

        if batch['weight'] is not None:
            mask = batch['weight'] == 0
        else:
            mask = batch['target'] == 0
        
        if predict:
            mask[:, self.encoder_length:] = True

        mus, sigs, _ = self.encoder(feat, embedded_labels=tgt_emb, mask=mask)
        
        if self.drop_variance:
            return mus.unsqueeze(-1)
        return torch.stack((mus, sigs), dim=-1)

    def predict(self, batch):
        preds = self.forward(batch, predict=True)
        preds = preds[:,self.encoder_length:, :]
        if self.drop_variance:
            return preds
        preds = torch.stack([preds[...,0] + preds[...,1] * torch.erfinv(2 * q - 1) * 1.4142135623730951 for q in self.quantiles], dim=-1)
        return preds
