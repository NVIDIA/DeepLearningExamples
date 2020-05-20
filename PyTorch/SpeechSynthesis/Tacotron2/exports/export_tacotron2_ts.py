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

import torch
import argparse
import sys
sys.path.append('./')
from inference import checkpoint_from_distributed, unwrap_distributed, load_and_setup_model

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--tacotron2', type=str, required=True,
                        help='full path to the Tacotron2 model checkpoint file')

    parser.add_argument('-o', '--output', type=str, default="trtis_repo/tacotron/1/model.pt",
                        help='filename for the Tacotron 2 TorchScript model')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with mixed precision')

    return parser


def main():

    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args = parser.parse_args()

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     amp_run=args.fp16, cpu_run=False,
                                     forward_is_infer=True)
    
    jitted_tacotron2 = torch.jit.script(tacotron2)

    torch.jit.save(jitted_tacotron2, args.output)
    

if __name__ == '__main__':
    main()

    
