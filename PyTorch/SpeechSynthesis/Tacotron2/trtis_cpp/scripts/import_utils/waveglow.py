#!/usr/bin/env python3
##
# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     # Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     # Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     # Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
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
# 



import pickle
import torch
from waveglow.model import WaveGlow

def split_cond_layers(model):
    for WN in model.WN:
        if hasattr(WN, "cond_layer"):
            n_layers = len(WN.res_skip_layers)
            conv_weights = WN.cond_layer.weight
            conv_bias = WN.cond_layer.bias
            conv_stride = WN.cond_layer.stride
            conv_dilation = WN.cond_layer.dilation
            conv_padding = WN.cond_layer.padding
            num_in_channels = conv_weights.size(1)
            num_out_channels = conv_weights.size(0)//n_layers
            kernel_size = conv_weights.size(2)
            WN.cond_layers = []
            for i in range(n_layers):
                layer = torch.nn.Conv1d(
                    in_channels=num_in_channels,
                    out_channels=num_out_channels,
                    kernel_size=kernel_size,
                    stride=conv_stride,
                    padding=conv_padding,
                    dilation=conv_dilation)
                layer.weight.data[:, :, :] = conv_weights.data[
                        i*num_out_channels:(i+1)*num_out_channels, :, :]
                layer.bias.data[:] = conv_bias.data[
                        i*num_out_channels:(i+1)*num_out_channels]
                layer = torch.nn.utils.weight_norm(layer, name='weight')
                WN.cond_layers.append(layer)
    return model



def load_waveglow(filename, waveglow_config):
    class RenamingUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'glow':
                module = 'waveglow.model'
            return super().find_class(module, name)

    class RenamingPickleModule:
        def load(self, f, *args, **kw_args):
            return self.Unpickler(f, *args, **kw_args).load()

        def Unpickler(self, f, **pickle_load_args):
            return RenamingUnpickler(f, **pickle_load_args)

    pickle_module = RenamingPickleModule()
    blob = torch.load(filename, pickle_module=pickle_module)

    if 'state_dict' in blob:
        waveglow = WaveGlow(**waveglow_config).cuda()
        state_dict = {}
        for key, value in blob["state_dict"].items():
            newKey = key
            if key.startswith("module."):
                newKey = key[len("module."):]
            state_dict[newKey] = value
        waveglow.load_state_dict(state_dict)
    else:
        waveglow = blob['model']

    waveglow = split_cond_layers(waveglow)
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()

    return waveglow
