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
from os.path import abspath, dirname
# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/'))
from tacotron2.model import Tacotron2
from waveglow.model import WaveGlow
import torch


def model_parser(model_name, parser, add_help=False):
    if model_name == 'Tacotron2':
        from tacotron2.arg_parser import tacotron2_parser
        return tacotron2_parser(parser, add_help)
    if model_name == 'WaveGlow':
        from waveglow.arg_parser import waveglow_parser
        return waveglow_parser(parser, add_help)
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


def get_model(model_name, model_config, cpu_run,
              uniform_initialize_bn_weight=False, forward_is_infer=False):
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
    else:
        raise NotImplementedError(model_name)

    if uniform_initialize_bn_weight:
        init_bn(model)

    if not cpu_run:
        model = model.cuda()
    return model


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
            decoder_no_early_stopping=args.decoder_no_early_stopping
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
    else:
        raise NotImplementedError(model_name)
