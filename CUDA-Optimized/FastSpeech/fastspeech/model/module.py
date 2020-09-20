# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from fastspeech.text_norm.symbols import symbols
from fastspeech.utils.nvtx import Nvtx
from fastspeech.utils.pytorch import to_device_async

try:
    import apex
except ImportError:
    ImportError('Required to install apex.')


class Bmm(Module):
    """ Required for manual fp16 casting. If not using amp_opt_level='O2', just use torch.bmm.
    """
    def forward(self, a, b):
        return torch.bmm(a, b)


class FFTBlocks(nn.Module):
    def __init__(self,
                 max_seq_len,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 name,
                 fused_layernorm=False,
                 ):

        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.fft_conv1_kernel = fft_conv1d_kernel
        self.fft_conv1d_padding = fft_conv1d_padding
        self.droupout = dropout
        self.fused_layernorm = fused_layernorm
        self.name = name

        super(FFTBlocks, self).__init__()

        n_position = max_seq_len + 1
        self.position = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            fused_layernorm=fused_layernorm,
            name="{}.layer_stack.{}".format(self.name, i),
        ) for i in range(n_layers)])

    def forward(self, seq, pos, return_attns=False, acts=None):

        slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=pos, seq_q=pos)  # (b, t, t)
        non_pad_mask = get_non_pad_mask(pos)  # (b, t, 1)

        # -- Forward
        pos_enc = self.position(pos)
        output = seq + pos_enc

        if acts is not None:
            acts["act.{}.add_pos_enc".format(self.name)] = output

        for i, layer in enumerate(self.layer_stack):
            output, slf_attn = layer(
                output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                acts=acts)
            if return_attns:
                slf_attn_list += [slf_attn]

            if acts is not None:
                acts['act.{}.layer_stack.{}'.format(self.name, i)] = output

        return output, non_pad_mask


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 name,
                 fused_layernorm=False):
        super(FFTBlock, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.fft_conv1_kernel = fft_conv1d_kernel
        self.fft_conv1d_padding = fft_conv1d_padding
        self.droupout = dropout
        self.name = name
        self.fused_layernorm = fused_layernorm

        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            name="{}.slf_attn".format(name),
            fused_layernorm=fused_layernorm)

        self.pos_ffn = PositionwiseFeedForward(
            d_in=d_model,
            d_hid=d_inner,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            name="{}.pos_ffn".format(name),
            fused_layernorm=fused_layernorm)

    @Nvtx("fftblock", enabled=False)
    def forward(self, input, non_pad_mask=None, slf_attn_mask=None, acts=None):
        output, slf_attn = self.slf_attn(
            input, mask=slf_attn_mask, acts=acts)

        output *= non_pad_mask.to(output.dtype)

        output = self.pos_ffn(output, acts=acts)
        output *= non_pad_mask.to(output.dtype)

        return output, slf_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout, name, fused_layernorm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.name = name

        d_out = d_k + d_k + d_v
        self.linear = nn.Linear(d_model, n_head * d_out)
        nn.init.xavier_normal_(self.linear.weight)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5),
            name="{}.scaled_dot".format(self.name))

        self.layer_norm = apex.normalization.FusedLayerNorm(
            d_model) if fused_layernorm else nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    @Nvtx("slf_attn", enabled=False)
    def forward(self, x, mask=None, acts=None):
        bs, seq_len, _ = x.size()

        residual = x

        with Nvtx("linear", enabled=False):
            d_out = self.d_k + self.d_k + self.d_v
            x = self.linear(x)  # (b, t, n_heads * h)

            if acts is not None:
                acts['act.{}.linear'.format(self.name)] = x

            x = x.view(bs, seq_len, self.n_head, d_out)  # (b, t, n_heads, h)
            x = x.permute(2, 0, 1, 3).contiguous().view(self.n_head * bs, seq_len, d_out)  # (n * b, t, h)

            q = x[..., :self.d_k]  # (n * b, t, d_k)
            k = x[..., self.d_k: 2*self.d_k]  # (n * b, t, d_k)
            v = x[..., 2*self.d_k:]  # (n * b, t, d_k)

        with Nvtx("mask repeat", enabled=False):
            mask = mask.repeat(self.n_head, 1, 1)  # (b, t, h) -> (n * b, t, h)

        with Nvtx("scaled dot", enabled=False):
            output, attn = self.attention(q, k, v, mask=mask, acts=acts)

        output = output.view(self.n_head, bs, seq_len, self.d_v)  # (n, b, t, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            bs, seq_len, self.n_head * self.d_v)  # (b, t, n * d_k)

        if acts is not None:
            acts['act.{}.scaled_dot'.format(self.name)] = output

        with Nvtx("fc", enabled=False):
            output = self.fc(output)

        with Nvtx("dropout", enabled=False):
            output = self.dropout(output)

        output += residual

        if acts is not None:
            acts['act.{}.residual'.format(self.name)] = output

        with Nvtx("layer norm", enabled=False):
            output = self.layer_norm(output)

        if acts is not None:
            acts['act.{}.ln'.format(self.name)] = output

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, name=None):
        super().__init__()

        self.temperature = temperature
        self.name = name

        self.bmm1 = Bmm()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.bmm2 = Bmm()

    @Nvtx("scaled_dot", enabled=False)
    def forward(self, q, k, v, mask=None, acts=None):

        with Nvtx("bmm1", enabled=False):
            attn = self.bmm1(q, k.transpose(1, 2))

        attn = attn / self.temperature

        with Nvtx("mask", enabled=False):
            if mask is not None:
                attn = attn.masked_fill(mask, -65504)

        with Nvtx("softmax", enabled=False):
            attn = self.softmax(attn)

        with Nvtx("dropout", enabled=False):
            attn = self.dropout(attn)

        with Nvtx("bmm2", enabled=False):
            output = self.bmm2(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self,
                 d_in,
                 d_hid,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 name,
                 fused_layernorm=False):
        super().__init__()

        self.name = name

        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel, padding=fft_conv1d_padding)

        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel, padding=fft_conv1d_padding)

        self.layer_norm = apex.normalization.FusedLayerNorm(
            d_in) if fused_layernorm else nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(dropout)

    @Nvtx("position wise", enabled=False)
    def forward(self, x, acts=None):
        residual = x

        output = x.transpose(1, 2)
        output = self.w_1(output)

        if acts is not None:
            acts['act.{}.conv1'.format(self.name)] = output

        output = F.relu(output)
        output = self.w_2(output)

        if acts is not None:
            acts['act.{}.conv2'.format(self.name)] = output

        output = output.transpose(1, 2)
        output = self.dropout(output)
        output += residual

        if acts is not None:
            acts['act.{}.residual'.format(self.name)] = output

        output = self.layer_norm(output)

        if acts is not None:
            acts['act.{}.ln'.format(self.name)] = output

        return output


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)  # (b, t)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # (b, t, t)

    return padding_mask


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).unsqueeze(-1)


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, input_size, duration_predictor_filter_size, duration_predictor_kernel_size, dropout, fused_layernorm=False):
        super(LengthRegulator, self).__init__()

        self.duration_predictor = DurationPredictor(
            input_size=input_size,
            filter_size=duration_predictor_filter_size,
            kernel=duration_predictor_kernel_size,
            dropout=dropout,
            fused_layernorm=fused_layernorm
        )

    @Nvtx("length regulator", enabled=False)
    def forward(self, input, input_mask, target=None, alpha=1.0):
        duration = self.duration_predictor(
            input, input_mask)
        # print(duration_predictor_output)

        if self.training:
            output, output_pos = self.get_output(
                input, target, alpha)
        else:
            duration = torch.clamp_min(torch.exp(duration) - 1, 0)
            output, output_pos = self.get_output(
                input, duration, alpha)

        return output, output_pos, duration

    def get_output(self, input, duration, alpha):
        output, output_pos = list(), list()
        # TODO: parallelize the loop.
        for i in range(input.size(0)):
            repeats = duration[i].float() * alpha
            with Nvtx("round #{}".format(i), enabled=False):
                repeats = torch.round(repeats).long()
            with Nvtx("repeat #{}".format(i), enabled=False):
                output.append(torch.repeat_interleave(
                    input[i], repeats, dim=0))
            output_pos.append(torch.from_numpy(
                np.indices((output[i].shape[0],))[0] + 1))
        output = pad_sequence(output, batch_first=True)
        output_pos = pad_sequence(output_pos, batch_first=True)

        with Nvtx("pos to gpu", enabled=False):
            output_pos = to_device_async(output_pos, device=output.device)

        return output, output_pos


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, input_size, filter_size, kernel, dropout, fused_layernorm=False):
        super(DurationPredictor, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel
        self.dropout = dropout

        self.conv1d_1 = nn.Conv1d(self.input_size,
                                  self.filter_size,
                                  kernel_size=self.kernel,
                                  padding=1)
        self.relu_1 = nn.ReLU()
        self.layer_norm_1 = apex.normalization.FusedLayerNorm(
            self.filter_size) if fused_layernorm else nn.LayerNorm(self.filter_size)

        self.dropout_1 = nn.Dropout(self.dropout)

        self.conv1d_2 = nn.Conv1d(self.filter_size,
                                  self.filter_size,
                                  kernel_size=self.kernel,
                                  padding=1)
        self.relu_2 = nn.ReLU()

        self.layer_norm_2 = apex.normalization.FusedLayerNorm(
            self.filter_size) if fused_layernorm else nn.LayerNorm(self.filter_size)

        self.dropout_2 = nn.Dropout(self.dropout)

        self.linear_layer = nn.Linear(self.filter_size, 1, bias=True)

    @Nvtx("duration predictor", enabled=False)
    def forward(self, input, input_mask):
        input *= input_mask.to(input.dtype)

        out = self.conv1d_1(input.transpose(1,2)).transpose(1,2)
        out = self.relu_1(out)
        out = self.layer_norm_1(out)
        out = self.dropout_1(out)

        out = self.conv1d_2(out.transpose(1,2)).transpose(1,2)
        out = self.relu_2(out)
        out = self.layer_norm_2(out)
        out = self.dropout_2(out)

        out = self.linear_layer(out)

        out *= input_mask.to(out.dtype)
        out = out.squeeze(-1)

        return out
