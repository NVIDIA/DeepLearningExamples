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

import argparse
import torch
from inference import load_and_setup_model


def parse_args(parser):
    parser.add_argument('--generator-name', type=str, required=True,
                        choices=('Tacotron2', 'FastPitch'), help='model name')
    parser.add_argument('--generator-checkpoint', type=str, required=True,
                        help='full path to the generator checkpoint file')
    parser.add_argument('-o', '--output', type=str, default="trtis_repo/tacotron/1/model.pt",
                        help='filename for the Tacotron 2 TorchScript model')
    parser.add_argument('--amp', action='store_true',
                        help='inference with AMP')
    return parser


def main():
    parser = argparse.ArgumentParser(description='Export models to TorchScript')
    parser = parse_args(parser)
    args = parser.parse_args()

    model = load_and_setup_model(
        args.generator_name, parser, args.generator_checkpoint,
        args.amp, device='cpu', forward_is_infer=True, polyak=False,
        jitable=True)
    
    torch.jit.save(torch.jit.script(model), args.output)
    

if __name__ == '__main__':
    main()

    
