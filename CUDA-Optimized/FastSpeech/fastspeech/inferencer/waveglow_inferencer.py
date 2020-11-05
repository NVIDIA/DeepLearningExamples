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

import sys

import torch

from fastspeech.utils.logging import tprint
from fastspeech.utils.pytorch import to_cpu_numpy, to_device_async
from fastspeech.inferencer.denoiser import Denoiser

from waveglow.model import WaveGlow
import argparse

def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

class WaveGlowInferencer(object):

    def __init__(self, ckpt_file, device='cuda', use_fp16=False, use_denoiser=False):
        self.ckpt_file = ckpt_file
        self.device = device
        self.use_fp16 = use_fp16
        self.use_denoiser = use_denoiser

        # model
        # sys.path.append('waveglow')

        from waveglow.arg_parser import parse_waveglow_args
        parser = parser = argparse.ArgumentParser()
        model_parser= parse_waveglow_args(parser)
        args, _ = model_parser.parse_known_args()
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
        self.model = WaveGlow(**model_config)

        state_dict = torch.load(self.ckpt_file, map_location=self.device)['state_dict']
        state_dict = unwrap_distributed(state_dict)
        self.model.load_state_dict(state_dict)

        self.model = to_device_async(self.model, self.device)

        self.model = self.model.remove_weightnorm(self.model)

        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()
        self.model = self.model

        if self.use_denoiser:
            self.denoiser = Denoiser(self.model, device=device)
            self.denoiser = to_device_async(self.denoiser, self.device)

            tprint('Using WaveGlow denoiser.')

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def infer(self, mels):
        if self.use_fp16:
            mels = mels.half()
        mels = to_device_async(mels, self.device)
        wavs = self.model.infer(mels, sigma=0.6)

        if self.use_denoiser:
            wavs = self.denoiser(wavs, strength=0.01)

        return wavs.float()