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

import torch
import argparse
import os
import sys
sys.path.append('./')

from tacotron2_common.utils import ParseFromConfigFile
from inference import load_and_setup_model

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

    parser.add_argument('--config-file', action=ParseFromConfigFile,
                         type=str, help='Path to configuration file')

    return parser


def export_onnx(parser, args):

    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    fp16_run=args.fp16, cpu_run=False,
                                    forward_is_infer=False)

    # 80 mel channels, 620 mel spectrograms ~ 7 seconds of speech
    mel = torch.randn(1, 80, 620).cuda()
    stride = 256 # value from waveglow upsample
    n_group = 8
    z_size2 = (mel.size(2)*stride)//n_group
    z = torch.randn(1, n_group, z_size2).cuda()

    if args.fp16:
        mel = mel.half()
        z = z.half()
    with torch.no_grad():
        # run inference to force calculation of inverses
        waveglow.infer(mel, sigma=args.sigma_infer)

        # export to ONNX
        if args.fp16:
            waveglow = waveglow.half()

        waveglow.forward = waveglow.infer_onnx

        opset_version = 12

        output_path = os.path.join(args.output, "waveglow.onnx")
        torch.onnx.export(waveglow, (mel, z), output_path,
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
