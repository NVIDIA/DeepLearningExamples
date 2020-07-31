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

from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn

from fastspeech.model.module import FFTBlocks, LengthRegulator
from fastspeech.utils.pytorch import to_device_async
from fastspeech.utils.nvtx import Nvtx
from torch.nn import functional as F
from fastspeech.utils.logging import tprint
from fastspeech.text_norm.symbols import symbols

class Fastspeech(nn.Module):
    """ FastSpeech """

    def __init__(self, 
                 max_seq_len, 
                 d_model,
                 phoneme_side_n_layer, 
                 phoneme_side_head, 
                 phoneme_side_conv1d_filter_size,
                 phoneme_side_output_size, 
                 mel_side_n_layer, 
                 mel_side_head, 
                 mel_side_conv1d_filter_size,
                 mel_side_output_size,
                 fft_conv1d_kernel, 
                 fft_conv1d_padding,
                 duration_predictor_filter_size, 
                 duration_predictor_kernel_size, 
                 dropout,
                 n_mels,
                 fused_layernorm=False):
        super(Fastspeech, self).__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.phoneme_side_n_layer = phoneme_side_n_layer
        self.phoneme_side_head = phoneme_side_head
        self.phoneme_side_conv1d_filter_size = phoneme_side_conv1d_filter_size
        self.phoneme_side_output_size = phoneme_side_output_size
        self.mel_side_n_layer = mel_side_n_layer
        self.mel_side_head = mel_side_head
        self.mel_side_conv1d_filter_size = mel_side_conv1d_filter_size
        self.mel_side_output_size = mel_side_output_size
        self.fft_conv1d_kernel = fft_conv1d_kernel
        self.fft_conv1d_padding = fft_conv1d_padding
        self.duration_predictor_filter_size = duration_predictor_filter_size
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.dropout = dropout
        self.n_mels = n_mels
        self.fused_layernorm = fused_layernorm
        self.n_phns = len(symbols)+1

        self.word_emb = nn.Embedding(
            self.n_phns, 
            d_model, 
            padding_idx=0)

        self.phoneme_side = FFTBlocks(
            max_seq_len=max_seq_len,
            n_layers=phoneme_side_n_layer,
            n_head=phoneme_side_head,
            d_k=64,
            d_v=64,
            d_model=d_model,
            d_inner=phoneme_side_conv1d_filter_size,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            name="phoneme_side",
            fused_layernorm=fused_layernorm
        )

        self.length_regulator = LengthRegulator(
            input_size=phoneme_side_output_size,
            duration_predictor_filter_size=duration_predictor_filter_size,
            duration_predictor_kernel_size=duration_predictor_kernel_size,
            dropout=dropout,
            fused_layernorm=fused_layernorm
        )

        self.mel_side = FFTBlocks(
            max_seq_len=max_seq_len,
            n_layers=mel_side_n_layer,
            n_head=mel_side_head,
            d_k=64,
            d_v=64,
            d_model=d_model,
            d_inner=mel_side_conv1d_filter_size,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout,
            name="mel_side",
            fused_layernorm=fused_layernorm            
        )

        self.mel_linear = nn.Linear(mel_side_output_size, n_mels, bias=True)

    def forward(self, seq, pos, duration_target=None, alpha=1.0, seq_output_len=None, use_fp16=False, acts=None):

        # Phoneme Embedding
        output = self.word_emb(seq)

        if acts is not None:
            acts["act.emb"] = output

        if use_fp16:
            output = output.half()

        # Phoneme Side FFT Blocks
        output, output_mask = self.phoneme_side(output, pos, acts=acts)

        if acts is not None:
            acts["act.phoneme_side.seq"] = output

        # Length Regulator
        output, pos, duration = self.length_regulator(
            output,
            output_mask,
            target=duration_target,
            alpha=alpha)

        if seq_output_len:
            output = F.pad(output, pad=(0, 0, 0, seq_output_len - output.size(1)))
            pos = F.pad(pos, pad=(0, seq_output_len - pos.size(1)))

        # length of output mel shouldn't exceed max_seq_len
        output = output[:, :self.max_seq_len]
        pos = pos[:, :self.max_seq_len]

        if acts is not None:
            acts["act.length_regulator.seq"] = output
            acts["act.length_regulator.dur"] = torch.round(duration)

        if self.training or output.bool().any():      
            # Mel Side FFT Blocks
            output, output_mask = self.mel_side(output, pos, acts=acts)

            if acts is not None:
                acts["act.mel_side.seq"] = output

            # Linear Layer
            output = self.mel_linear(output)

            if acts is not None:
                acts["out.seq_mask"] = output_mask
                acts["out.seq"] = output
        else:
            # seq length could be zero, in case duration predictor outputs all zeros.
            # In this case, skip feed-forwarding.
            tprint("Duration Predictor outputs all zeros. Output will be zero length.")
            output_shape = (output.size(0), 0, output_mask.size(2))
            output = torch.zeros(size=(output_shape))
            output_mask = torch.ones(size=(output_shape))

        if torch.cuda.device_count() > 1:
            # In a multi-gpu setting, all output mels from devices must have the same length.
            # otherwise, an error occurs in process of gathering output.
            if not seq_output_len:
                seq_output_len = self.max_seq_len
            padding = (0, 0, 0, seq_output_len - output.size(1))

            output = F.pad(output, padding)
            output = output[:, :seq_output_len, :]

            output_mask = F.pad(output_mask, padding)
            output_mask = output_mask[:, :seq_output_len, :]          

        return output, output_mask, duration
