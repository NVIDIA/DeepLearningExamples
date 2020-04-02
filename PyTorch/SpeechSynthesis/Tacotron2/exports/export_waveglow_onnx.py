# *****************************************************************************
#  Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import types
import torch
import argparse

import sys
sys.path.append('./')

from inference import checkpoint_from_distributed, unwrap_distributed, load_and_setup_model

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--waveglow', type=str, required=True,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory for the exported WaveGlow ONNX model')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with AMP')
    parser.add_argument('-s', '--sigma-infer', default=0.6, type=float)

    return parser


def convert_convinv_1d_to_2d(convinv):
    """
    Takes an invertible 1x1 1-d convolution and returns a 2-d convolution that does
    the inverse
    """
    conv2d = torch.nn.Conv2d(convinv.W_inverse.size(1),
                             convinv.W_inverse.size(0),
                             1, bias=False)
    conv2d.weight.data[:,:,:,0] = convinv.W_inverse.data
    return conv2d


def convert_conv_1d_to_2d(conv1d):
    conv2d = torch.nn.Conv2d(conv1d.weight.size(1),
                             conv1d.weight.size(0),
                             (conv1d.weight.size(2), 1),
                             stride=(conv1d.stride[0], 1),
                             dilation=(conv1d.dilation[0], 1),
                             padding=(conv1d.padding[0], 0))
    conv2d.weight.data[:,:,:,0] = conv1d.weight.data
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
    upsample.weight.data[:,:,:,0] = glow.upsample.weight.data
    upsample.bias.data = glow.upsample.bias.data
    glow.upsample = upsample.cuda()

    # Convert WN to 2d
    for WN in glow.WN:
        convert_WN_1d_to_2d_(WN)

    # Convert invertible conv to 2d
    for i in range(len(glow.convinv)):
        glow.convinv[i] = convert_convinv_1d_to_2d(glow.convinv[i])

    glow.cuda()


def infer_onnx(self, spect, z, sigma=0.9):

    spect = self.upsample(spect)
    # trim conv artifacts. maybe pad spec to kernel multiple
    time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
    spect = spect[:, :, :-time_cutoff]

    length_spect_group = spect.size(2)//8
    mel_dim = 80
    batch_size = spect.size(0)

    spect = torch.squeeze(spect, 3)
    spect = spect.view((batch_size, mel_dim, length_spect_group, self.n_group))
    spect = spect.permute(0, 2, 1, 3)
    spect = spect.contiguous()
    spect = spect.view((batch_size, length_spect_group, self.n_group*mel_dim))
    spect = spect.permute(0, 2, 1)
    spect = torch.unsqueeze(spect, 3)
    spect = spect.contiguous()

    audio = z[:, :self.n_remaining_channels, :, :]
    z = z[:, self.n_remaining_channels:self.n_group, :, :]
    audio = sigma*audio

    for k in reversed(range(self.n_flows)):
        n_half = int(audio.size(1) / 2)
        audio_0 = audio[:, :n_half, :, :]
        audio_1 = audio[:, n_half:(n_half+n_half), :, :]

        output = self.WN[k]((audio_0, spect))
        s = output[:, n_half:(n_half+n_half), :, :]
        b = output[:, :n_half, :, :]
        audio_1 = (audio_1 - b) / torch.exp(s)
        audio = torch.cat([audio_0, audio_1], 1)

        audio = self.convinv[k](audio)

        if k % self.n_early_every == 0 and k > 0:
            audio = torch.cat((z[:, :self.n_early_size, :, :], audio), 1)
            z = z[:, self.n_early_size:self.n_group, :, :]

    audio = torch.squeeze(audio, 3)
    audio = audio.permute(0,2,1).contiguous().view(batch_size, (length_spect_group * self.n_group))

    return audio


def export_onnx(parser, args):

    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    args.fp16, forward_is_infer=False)

    # 80 mel channels, 620 mel spectrograms ~ 7 seconds of speech
    mel = torch.randn(1, 80, 620).cuda()
    stride = 256 # value from waveglow upsample
    n_group = 8
    z_size2 = (mel.size(2)*stride)//n_group
    z = torch.randn(1, n_group, z_size2, 1).cuda()

    if args.fp16:
        mel = mel.half()
        z = z.half()
    with torch.no_grad():
        # run inference to force calculation of inverses
        waveglow.infer(mel, sigma=args.sigma_infer)

        # export to ONNX
        convert_1d_to_2d_(waveglow)
        if args.fp16:
            waveglow = waveglow.half()

        fType = types.MethodType
        waveglow.forward = fType(infer_onnx, waveglow)

        mel = mel.unsqueeze(3)

        opset_version = 10

        torch.onnx.export(waveglow, (mel, z), args.output+"/"+"waveglow.onnx",
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["mel", "z"],
                          output_names=["audio"],
                          dynamic_axes={"mel":   {0: "batch_size", 2: "mel_seq"},
                                        "z":     {0: "batch_size", 2: "z_seq"},
                                        "audio": {0: "batch_size", 1: "audio_seq"}})


def main():

    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    export_onnx(parser, args)

if __name__ == '__main__':
    main()
