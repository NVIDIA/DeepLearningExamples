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

import json
import sys
import onnx
import numpy as np
from scipy.io.wavfile import write
import argparse
import torch

args = None


def convert_conv_1d_to_2d(conv1d):
    conv2d = torch.nn.Conv2d(conv1d.weight.size(1),
                             conv1d.weight.size(0),
                             (conv1d.weight.size(2), 1),
                             stride=(conv1d.stride[0], 1),
                             dilation=(conv1d.dilation[0], 1),
                             padding=(conv1d.padding[0], 0))
    conv2d.weight.data[:, :, :, 0] = conv1d.weight.data
    conv2d.bias.data = conv1d.bias.data
    return conv2d


def convert_WN_1d_to_2d_(WN):
    """
    Modifies the WaveNet like affine coupling layer in-place to use 2-d convolutions
    """
    WN.start = convert_conv_1d_to_2d(WN.start)
    WN.end = convert_conv_1d_to_2d(WN.end)

    for i in range(len(WN.in_layers)):
        WN.in_layers[i] = convert_conv_1d_to_2d(WN.in_layers[i])

    for i in range(len(WN.res_skip_layers)):
        WN.res_skip_layers[i] = convert_conv_1d_to_2d(WN.res_skip_layers[i])

    for i in range(len(WN.res_skip_layers)):
        WN.cond_layers[i] = convert_conv_1d_to_2d(WN.cond_layers[i])


def convert_convinv_1d_to_2d(convinv):
    """
    Takes an invertible 1x1 1-d convolution and returns a 2-d convolution that does
    the inverse
    """
    conv2d = torch.nn.Conv2d(convinv.W_inverse.size(1),
                             convinv.W_inverse.size(0),
                             1, bias=False)
    conv2d.weight.data[:, :, :, 0] = convinv.W_inverse.data
    return conv2d


def convert_1d_to_2d_(glow):
    """
    Caffe2 and TensorRT don't seem to support 1-d convolutions or properly
    convert ONNX exports with 1d convolutions to 2d convolutions yet, so we
    do the conversion to 2-d convolutions before ONNX export
    """
    # Convert upsample to 2d
    upsample = torch.nn.ConvTranspose2d(glow.upsample.weight.size(0),
                                        glow.upsample.weight.size(1),
                                        (glow.upsample.weight.size(2), 1),
                                        stride=(glow.upsample.stride[0], 1))
    upsample.weight.data[:, :, :, 0] = glow.upsample.weight.data
    upsample.bias.data = glow.upsample.bias.data
    glow.upsample = upsample

    # Convert WN to 2d
    for WN in glow.WN:
        convert_WN_1d_to_2d_(WN)

    # Convert invertible conv to 2d
    for i in range(len(glow.convinv)):
        glow.convinv[i] = convert_convinv_1d_to_2d(glow.convinv[i])


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    in_act = input_a+input_b
    in_left = in_act[:, 0:n_channels, :, :]
    in_right = in_act[:, n_channels:2*n_channels, :, :]
    t_act = torch.tanh(in_left)
    s_act = torch.sigmoid(in_right)
    acts = t_act * s_act
    return acts


def WN_forward(self, forward_input):
    """
    This is a forward replacement for the WN forward.  This is required because
    the code was written for 1d convs which isn't yet supported from ONNX
    exports.
    """
    audio, spect = forward_input
    audio = self.start(audio)

    for i in range(self.n_layers):
        acts = fused_add_tanh_sigmoid_multiply(
            self.in_layers[i](audio),
            self.cond_layers[i](spect),
            self.n_channels)

        res_skip_acts = self.res_skip_layers[i](acts)
        if i < self.n_layers - 1:
            audio = res_skip_acts[:, 0:self.n_channels, :, :] + audio
            skip_acts = res_skip_acts[:,
                                      self.n_channels:2*self.n_channels, :, :]
        else:
            skip_acts = res_skip_acts

        if i == 0:
            output = skip_acts
        else:
            output = skip_acts + output
    return self.end(output)


def infer_o(self, spect, z):
    """
    In order to for the trace to work running through ONNX with 2d convolutions
    we need to overwrite the forward method.  All shape information is
    pre-calculated so ONNX doesn't export "Dynamic" outputs which are not yet
    suported by TensorRT
    """
    batch_size = spect.size(0)
    spect = spect.permute(0, 3, 2, 1).contiguous()

    spect = self.upsample(spect)
    spect = torch.squeeze(spect, 3)
    spect = spect.view(batch_size, self.upsample_weight_size,  self.length_spect_group, self.n_group)
    spect = spect.permute(0, 2, 1, 3)
    spect = spect.contiguous()
    spect = spect.view(batch_size, self.length_spect_group, self.upsample_weight_size*self.n_group)
    spect = spect.permute(0, 2, 1)
    spect = torch.unsqueeze(spect, 3)
    spect = spect.contiguous()

    audio = z[:, :self.n_remaining_channels, :, :]
    z = z[:, self.n_remaining_channels:self.n_group, :, :]

    for k in reversed(range(self.n_flows)):
        n_half = self.n_halves[k]
        audio_0 = audio[:, 0:n_half, :, :]
        audio_1 = audio[:, n_half:2*n_half, :, :]

        output = self.WN[k]((audio_0, spect))
        s = output[:, n_half:2*n_half, :, :]
        b = output[:, 0:n_half, :, :]
        audio_1 = (audio_1 - b)/torch.exp(s)
        audio_0 = audio_0.expand(audio_1.size(0), audio_0.size(1),
                                 audio_0.size(2), audio_0.size(3))
        audio = torch.cat([audio_0, audio_1], 1)

        audio = self.convinv[k](audio)

        if k % self.n_early_every == 0 and k > 0:
            zb = z[:, 0:self.n_early_size, :, :].expand(audio.size(0),
                    self.n_early_size, z.size(2), z.size(3))
            audio = torch.cat((zb, audio), 1)
            z = z[:, self.n_early_size:self.n_group -
                  self.n_remaining_channels, :, :]

    audio = torch.squeeze(audio, 3)
    audio = audio.permute(0, 2, 1).contiguous().view(
        audio.size(0), (self.length_spect_group * self.n_group))
    return audio


def main(waveglow_path, output_path, batch_size, length_mels):
    """
    Takes a waveglow model, a batch size, and a length in mels about outputs a static
    ONNX representation using 2D convoultions
    """
    torch.manual_seed(0)

    model = load_waveglow(waveglow_path, waveglow_config)

    length_spect = length_mels
    length_samples = 768 + 256*length_spect

    model.upsample_weight_size = model.upsample.weight.size(0)

    spect = torch.cuda.FloatTensor(
        batch_size, model.upsample_weight_size, length_spect).normal_()
    spect = torch.autograd.Variable(spect.cuda(), requires_grad=False)

    # Run inference because it forces inverses to be calculated
    with torch.no_grad():
        test_out1 = model.infer(spect)
    assert(length_samples % model.n_group == 0)

    model.length_spect_group = int(length_samples / model.n_group)

    # Pre-calculating the sizes of noise to use so it's not dynamic
    n_halves = []
    n_half = int(model.n_remaining_channels/2)
    for k in reversed(range(model.n_flows)):
        n_halves.append(n_half)

        if k % model.n_early_every == 0 and k > 0:
            n_half = n_half + int(model.n_early_size/2)
    n_halves.reverse()
    model.n_halves = n_halves

    spect = torch.cuda.FloatTensor(
        batch_size, 1, length_spect, model.upsample.weight.size(0)).normal_()
    z = torch.cuda.FloatTensor(
        1, model.n_group, model.length_spect_group, 1).normal_()
    spect = torch.autograd.Variable(spect.cuda(), requires_grad=False)
    z = torch.autograd.Variable(z, requires_grad=False)

    # Replace old forward with inference
    glow.WaveGlow.forward = infer_o
    #glow.WN.forward = WN_forward

    # Convert whole model to 2d convolutions
    convert_1d_to_2d_(model)
    model.cuda()

    # Get output for comparison with Caffe2
    with torch.no_grad():
        test_out2 = model(spect, z)

    # Export model
    torch.onnx.export(model, (spect, z),
            output_path,
            dynamic_axes={'spect': {0: 'batch_size'},
                          'audio': {0: 'batch_size'}},
            input_names=['spect', 'z'],
            output_names=['audio'],
            opset_version=10,
            verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model',
                        required=True)
    parser.add_argument('-W', '--tacotron2_home', help='Path to DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2 directory.',
                        required=True)
    parser.add_argument('-o', "--onnx_path",
                        help="Path to output ONNX file", required=True)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--length_mels", default=160, type=int)

    # add wave glow arguments
    waveglow = parser.add_argument_group("WaveGlow parameters")
    waveglow.add_argument('--n-mel-channels', default=80, type=int,
                          help='Number of bins in mel-spectrograms')

    # glow parameters
    waveglow.add_argument('--flows', default=12, type=int,
                          help='Number of steps of flow')
    waveglow.add_argument('--groups', default=8, type=int,
                          help='Number of samples in a group processed by the steps of flow')
    waveglow.add_argument('--early-every', default=4, type=int,
                          help='Determines how often (i.e., after how many coupling layers) \
                        a number of channels (defined by --early-size parameter) are output\
                        to the loss function')
    waveglow.add_argument('--early-size', default=2, type=int,
                          help='Number of channels output to the loss function')
    waveglow.add_argument('--sigma', default=1.0, type=float,
                          help='Standard deviation used for sampling from Gaussian')
    waveglow.add_argument('--segment-length', default=4000, type=int,
                          help='Segment length (audio samples) processed per iteration')

    # wavenet parameters
    wavenet = waveglow.add_argument_group('WaveNet parameters')
    wavenet.add_argument('--wn-kernel-size', default=3, type=int,
                         help='Kernel size for dialted convolution in the affine coupling layer (WN)')
    wavenet.add_argument('--wn-channels', default=256, type=int,
                         help='Number of channels in WN')
    wavenet.add_argument('--wn-layers', default=8, type=int,
                         help='Number of layers in WN')

    args = parser.parse_args()

    # do imports as needed
    sys.path.append(args.tacotron2_home)

    import waveglow.model as glow
    from import_utils import load_waveglow

    global waveglow_config
    waveglow_config = {
        "n_mel_channels": args.n_mel_channels,
        "n_flows": args.flows,
        "n_group": args.groups,
        "n_early_every": args.early_every,
        "n_early_size": args.early_size,
        "WN_config": {
            "n_layers": args.wn_layers,
            "kernel_size": args.wn_kernel_size,
            "n_channels": args.wn_channels
        }
    }

    main(args.waveglow_path, args.onnx_path, args.batch_size, args.length_mels)
