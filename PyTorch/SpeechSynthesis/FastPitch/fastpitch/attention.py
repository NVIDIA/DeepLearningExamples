# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Invertible1x1ConvLUS(torch.nn.Module):
    def __init__(self, c):
        super(Invertible1x1ConvLUS, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        W, _ = torch.linalg.qr(torch.randn(c, c))
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.lu(W))

        self.register_buffer('p', p)
        # diagonals of lower will always be 1s anyway
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c, c))
        self.register_buffer('lower_diag', lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))

    def forward(self, z, reverse=False):
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z, log_det_W


class ConvAttention(torch.nn.Module):
    def __init__(self, n_mel_channels=80, n_speaker_dim=128,
                 n_text_channels=512, n_att_channels=80, temperature=1.0,
                 n_mel_convs=2, align_query_enc_type='3xconv',
                 use_query_proj=True):
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        self.att_scaling_factor = np.sqrt(n_att_channels)
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        self.attn_proj = torch.nn.Conv2d(n_att_channels, 1, kernel_size=1)
        self.align_query_enc_type = align_query_enc_type
        self.use_query_proj = bool(use_query_proj)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels,
                     n_text_channels * 2,
                     kernel_size=3,
                     bias=True,
                     w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2,
                     n_att_channels,
                     kernel_size=1,
                     bias=True))

        self.align_query_enc_type = align_query_enc_type

        if align_query_enc_type == "inv_conv":
            self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
        elif align_query_enc_type == "3xconv":
            self.query_proj = nn.Sequential(
                ConvNorm(n_mel_channels,
                         n_mel_channels * 2,
                         kernel_size=3,
                         bias=True,
                         w_init_gain='relu'),
                torch.nn.ReLU(),
                ConvNorm(n_mel_channels * 2,
                         n_mel_channels,
                         kernel_size=1,
                         bias=True),
                torch.nn.ReLU(),
                ConvNorm(n_mel_channels,
                         n_att_channels,
                         kernel_size=1,
                         bias=True))
        else:
            raise ValueError("Unknown query encoder type specified")

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model to run data through
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original,
            unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def encode_query(self, query, query_lens):
        query = query.permute(2, 0, 1)  # seq_len, batch, feature dim
        lens, ids = torch.sort(query_lens, descending=True)
        original_ids = [0] * lens.size(0)
        for i in range(len(ids)):
            original_ids[ids[i]] = i

        query_encoded = self.run_padded_sequence(ids, original_ids, lens,
                                                 query, self.query_lstm)
        query_encoded = query_encoded.permute(1, 2, 0)
        return query_encoded

    def forward(self, queries, keys, query_lens, mask=None, key_lens=None,
                keys_encoded=None, attn_prior=None):
        """Attention mechanism for flowtron parallel
        Unlike in Flowtron, we have no restrictions such as causality etc,
        since we only need this during training.

        Args:
            queries (torch.tensor): B x C x T1 tensor
                (probably going to be mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries
                (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                Final dim T2 should sum to 1
        """
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2

        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        if self.use_query_proj:
            if self.align_query_enc_type == "inv_conv":
                queries_enc, log_det_W = self.query_proj(queries)
            elif self.align_query_enc_type == "3xconv":
                queries_enc = self.query_proj(queries)
                log_det_W = 0.0
            else:
                queries_enc, log_det_W = self.query_proj(queries)
        else:
            queries_enc, log_det_W = queries, 0.0

        # different ways of computing attn,
        # one is isotopic gaussians (per phoneme)
        # Simplistic Gaussian Isotopic Attention

        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2
        # compute log likelihood from a gaussian
        attn = -0.0005 * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None]+1e-8)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2),
                                   -float("inf"))

        attn = self.softmax(attn)  # Softmax along T2
        return attn, attn_logprob
