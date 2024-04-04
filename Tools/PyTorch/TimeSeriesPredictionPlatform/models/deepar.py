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

import torch
import torch.nn as nn
from .tft_pyt.modeling import LazyEmbedding

class DeepAR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder_length = config.encoder_length
        self.register_buffer('quantiles', torch.FloatTensor(config.quantiles), persistent=False)
        self.use_embedding = self.config.use_embedding

        if self.config.use_embedding:
            ### New Embedding
            # DeepAR can't currenty work with observed data
            config.num_historic_vars -= len(config.temporal_observed_categorical_inp_lens)
            config.num_historic_vars -= config.temporal_observed_continuous_inp_size
            config.temporal_observed_categorical_inp_lens = []
            config.temporal_observed_continuous_inp_size = 0
            _config = config.copy()
            _config.hidden_size = self.config.embedding_dim
            self.embedding_v2 = LazyEmbedding(_config)
            inp_size = (config.num_static_vars + config.num_historic_vars) * config.embedding_dim
        else:
            self.embedding = nn.ModuleList([
                nn.Embedding(n, config.embedding_dim) 
                for n in config.static_categorical_inp_lens + config.temporal_known_categorical_inp_lens
                ])

            inp_size = config.temporal_known_continuous_inp_size + len(self.embedding) * config.embedding_dim + 1 # +1 for target

        self.lstm = nn.LSTM(input_size=inp_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            bias=True,
                            batch_first=True,
                            dropout=config.dropout)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(config.hidden_size * config.num_layers, 1)
        self.distribution_presigma = nn.Linear(config.hidden_size * config.num_layers, 1)
        self.distribution_sigma = nn.Softplus()

    def _roll_data(x):
        if x is None:
            return None
        x = torch.roll(x, 1, 1)
        x[:,0] = 0
        return x
    def forward(self, batch):
        if self.use_embedding:
            return self._forward_v2(batch)
        else:
            return self._forward_v1(batch)


    def _forward_v2(self, batch):
        batch = batch.copy() # shallow copy to replace observables in this scope
        batch['target'] = DeepAR._roll_data(batch['target'])
        batch['weight'] = DeepAR._roll_data(batch['weight'])
        batch['o_cat'] = None
        batch['o_cont'] = None

        emb = self.embedding_v2(batch)
        emb = [x for x in emb if x is not None]
        emb[0] = emb[0].unsqueeze(1).expand(emb[0].shape[0], emb[1].shape[1], *emb[0].shape[1:])
        emb = torch.cat(emb, axis=-2)
        emb = emb.view(*emb.shape[:-2], -1)

        state = None
        mus = []
        sigs = []
        for t in range(emb.shape[1]):
            zero_index = (batch['target'][:, t, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                _x = torch.matmul(mu[zero_index].unsqueeze(-1), self.embedding_v2.t_tgt_embedding_vectors)
                _x = _x + self.embedding_v2.t_tgt_embedding_bias
                emb[zero_index, t, -self.config.embedding_dim:] = _x  # target embedding is the last to be concatenated
            mu, sigma, state = self._forward_ar(emb[:,t].unsqueeze(1), state)

            mus.append(mu)
            sigs.append(sigma)

        mus = torch.stack(mus, dim=1)
        sigs = torch.stack(sigs, dim=1)

        return torch.stack((mus, sigs), dim=-1)

    def _forward_v1(self, batch):
        cat = torch.cat([batch['s_cat'], batch['k_cat']], dim=-1).permute(2,0,1)
        emb = torch.cat([e(t) for e, t in zip(self.embedding, cat)], dim=-1)
        target = torch.roll(batch['target'], 1, 1)
        target[:, 0] = 0
        x = torch.cat((target, batch['k_cont'], emb), dim=-1)

        state = None
        mus = []
        sigs = []
        for t in range(x.shape[1]):
            zero_index = (x[:, t, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                x[zero_index, t, 0] = mu[zero_index]

            mu, sigma, state = self._forward_ar(x[:, t].unsqueeze(1), state)

            mus.append(mu)
            sigs.append(sigma)

        mus = torch.stack(mus, dim=1)
        sigs = torch.stack(sigs, dim=1)

        return torch.stack((mus, sigs), dim=-1)

    def _forward_ar(self, x, state):
        output, state = self.lstm(x, state)
        hidden = state[0]
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive
        return torch.squeeze(mu), torch.squeeze(sigma), state

    def predict(self, batch):
        preds = self.forward(batch)
        preds = preds[:, self.encoder_length:, :]
        preds = torch.stack([preds[..., 0] + preds[..., 1] * torch.erfinv(2 * q - 1) * 1.4142135623730951 for q in self.quantiles], dim=-1)
        return preds
