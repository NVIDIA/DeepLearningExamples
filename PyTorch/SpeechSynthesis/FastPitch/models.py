# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from typing import Optional
from os.path import abspath, dirname

import torch

# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/'))
from fastpitch.model import FastPitch as _FastPitch
from fastpitch.model_jit import FastPitch as _FastPitchJIT
from tacotron2.model import Tacotron2
from waveglow.model import WaveGlow


def parse_model_args(model_name, parser, add_help=False):
    if model_name == 'Tacotron2':
        from tacotron2.arg_parser import parse_tacotron2_args
        return parse_tacotron2_args(parser, add_help)
    if model_name == 'WaveGlow':
        from waveglow.arg_parser import parse_waveglow_args
        return parse_waveglow_args(parser, add_help)
    elif model_name == 'FastPitch':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    else:
        raise NotImplementedError(model_name)


def batchnorm_to_float(module):
    """Converts batch norm to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def init_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)


def get_model(model_name, model_config, device,
              uniform_initialize_bn_weight=False, forward_is_infer=False,
              jitable=False):
    """ Code chooses a model based on name"""
    model = None
    if model_name == 'Tacotron2':
        if forward_is_infer:
            class Tacotron2__forward_is_infer(Tacotron2):
                def forward(self, inputs, input_lengths):
                    return self.infer(inputs, input_lengths)
            model = Tacotron2__forward_is_infer(**model_config)
        else:
            model = Tacotron2(**model_config)

    elif model_name == 'WaveGlow':
        if forward_is_infer:
            class WaveGlow__forward_is_infer(WaveGlow):
                def forward(self, spect, sigma=1.0):
                    return self.infer(spect, sigma)
            model = WaveGlow__forward_is_infer(**model_config)
        else:
            model = WaveGlow(**model_config)

    elif model_name == 'FastPitch':

        if forward_is_infer:

            if jitable:
                class FastPitch__forward_is_infer(_FastPitchJIT):
                    def forward(self, inputs, input_lengths, pace: float = 1.0,
                                dur_tgt: Optional[torch.Tensor] = None,
                                pitch_tgt: Optional[torch.Tensor] = None):
                        return self.infer(inputs, input_lengths, pace=pace,
                                          dur_tgt=dur_tgt, pitch_tgt=pitch_tgt)
            else:
                class FastPitch__forward_is_infer(_FastPitch):
                    def forward(self, inputs, input_lengths, pace: float = 1.0,
                                dur_tgt: Optional[torch.Tensor] = None,
                                pitch_tgt: Optional[torch.Tensor] = None,
                                pitch_transform=None):
                        return self.infer(inputs, input_lengths, pace=pace,
                                          dur_tgt=dur_tgt, pitch_tgt=pitch_tgt,
                                          pitch_transform=pitch_transform)

            model = FastPitch__forward_is_infer(**model_config)
        else:
            model = _FastPitch(**model_config)

    else:
        raise NotImplementedError(model_name)

    if uniform_initialize_bn_weight:
        init_bn(model)

    return model.to(device)


def get_model_config(model_name, args):
    """ Code chooses a model based on name"""
    if model_name == 'Tacotron2':
        model_config = dict(
            # optimization
            mask_padding=args.mask_padding,
            # audio
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=args.n_symbols,
            symbols_embedding_dim=args.symbols_embedding_dim,
            # encoder
            encoder_kernel_size=args.encoder_kernel_size,
            encoder_n_convolutions=args.encoder_n_convolutions,
            encoder_embedding_dim=args.encoder_embedding_dim,
            # attention
            attention_rnn_dim=args.attention_rnn_dim,
            attention_dim=args.attention_dim,
            # attention location
            attention_location_n_filters=args.attention_location_n_filters,
            attention_location_kernel_size=args.attention_location_kernel_size,
            # decoder
            n_frames_per_step=args.n_frames_per_step,
            decoder_rnn_dim=args.decoder_rnn_dim,
            prenet_dim=args.prenet_dim,
            max_decoder_steps=args.max_decoder_steps,
            gate_threshold=args.gate_threshold,
            p_attention_dropout=args.p_attention_dropout,
            p_decoder_dropout=args.p_decoder_dropout,
            # postnet
            postnet_embedding_dim=args.postnet_embedding_dim,
            postnet_kernel_size=args.postnet_kernel_size,
            postnet_n_convolutions=args.postnet_n_convolutions,
            decoder_no_early_stopping=args.decoder_no_early_stopping,
        )
        return model_config
    elif model_name == 'WaveGlow':
        model_config = dict(
            n_mel_channels=args.n_mel_channels,
            n_flows=args.flows,
            n_group=args.groups,
            n_early_every=args.early_every,
            n_early_size=args.early_size,
            WN_config=dict(
                n_layers=args.wn_layers,
                kernel_size=args.wn_kernel_size,
                n_channels=args.wn_channels
            )
        )
        return model_config
    elif model_name == 'FastPitch':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            max_seq_len=args.max_seq_len,
            # symbols
            n_symbols=args.n_symbols,
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
        )
        return model_config

    else:
        raise NotImplementedError(model_name)
