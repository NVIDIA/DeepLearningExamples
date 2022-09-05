# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from typing import Optional

import torch
from torch import nn as nn

from common import filter_warnings
from fastpitch.model import TemporalPredictor
from fastpitch.transformer_jit import FFTransformer


def regulate_len(durations, enc_out, pace: float = 1.0,
                 mel_max_len: Optional[int] = None):
    """If target=None, then predicted durations are applied"""
    reps = torch.round(durations.float() / pace).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    bsz, _, hid = enc_out.size()

    reps_padded = torch.cat([reps, (max_len - dec_lens)[:, None]], dim=1)
    pad_vec = torch.zeros(bsz, 1, hid, dtype=enc_out.dtype,
                          device=enc_out.device)

    enc_rep = torch.cat([enc_out, pad_vec], dim=1)
    enc_rep = torch.repeat_interleave(
        enc_rep.view(-1, hid), reps_padded.view(-1), dim=0
    ).view(bsz, -1, hid)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens


class FastPitchJIT(nn.Module):
    __constants__ = ['energy_conditioning']
    def __init__(self, n_mel_channels, n_symbols, padding_idx,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 p_dur_predictor_dropout, dur_predictor_n_layers,
                 pitch_predictor_kernel_size, pitch_predictor_filter_size,
                 p_pitch_predictor_dropout, pitch_predictor_n_layers,
                 pitch_embedding_kernel_size,
                 energy_conditioning,
                 energy_predictor_kernel_size, energy_predictor_filter_size,
                 p_energy_predictor_dropout, energy_predictor_n_layers,
                 energy_embedding_kernel_size,
                 n_speakers, speaker_emb_weight, pitch_conditioning_formants=1):
        super(FastPitchJIT, self).__init__()

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx)

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )

        self.pitch_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=pitch_predictor_filter_size,
            kernel_size=pitch_predictor_kernel_size,
            dropout=p_pitch_predictor_dropout, n_layers=pitch_predictor_n_layers,
            n_predictions=pitch_conditioning_formants
        )

        self.pitch_emb = nn.Conv1d(
            pitch_conditioning_formants, symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size,
            padding=int((pitch_embedding_kernel_size - 1) / 2))

        # Store values precomputed for training data within the model
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.energy_conditioning = energy_conditioning
        if energy_conditioning:
            self.energy_predictor = TemporalPredictor(
                in_fft_output_size,
                filter_size=energy_predictor_filter_size,
                kernel_size=energy_predictor_kernel_size,
                dropout=p_energy_predictor_dropout,
                n_layers=energy_predictor_n_layers,
                n_predictions=1
            )

            self.energy_emb = nn.Conv1d(
                1, symbols_embedding_dim,
                kernel_size=energy_embedding_kernel_size,
                padding=int((energy_embedding_kernel_size - 1) / 2))

        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)

        # skip self.attention (used only in training)

    def infer(self, inputs, pace: float = 1.0,
              dur_tgt: Optional[torch.Tensor] = None,
              pitch_tgt: Optional[torch.Tensor] = None,
              energy_tgt: Optional[torch.Tensor] = None,
              speaker: int = 0):

        if self.speaker_emb is None:
            spk_emb = None
        else:
            speaker = (torch.ones(inputs.size(0)).long().to(inputs.device)
                       * speaker)
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, 100.0)

        # Pitch over chars
        pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)

        if pitch_tgt is None:
            pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
        else:
            pitch_emb = self.pitch_emb(pitch_tgt).transpose(1, 2)

        enc_out = enc_out + pitch_emb

        # Predict energy
        if self.energy_conditioning:

            if energy_tgt is None:
                energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)
                energy_emb = self.energy_emb(energy_pred.unsqueeze(1)).transpose(1, 2)
            else:
                energy_pred = None
                energy_emb = self.energy_emb(energy_tgt).transpose(1, 2)

            enc_out = enc_out + energy_emb
        else:
            energy_pred = None

        len_regulated, dec_lens = regulate_len(
            dur_pred if dur_tgt is None else dur_tgt,
            enc_out, pace, mel_max_len=None)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        return mel_out, dec_lens, dur_pred, pitch_pred, energy_pred
