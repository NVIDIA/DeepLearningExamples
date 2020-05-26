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

import argparse
import json
import time

import torch
import numpy as np

import dllogger as DLLogger
from apex import amp
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

import models
from inference import load_and_setup_model, MeasureTime


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--amp-run', action='store_true',
                        help='inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=1)
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    return parser


def main():
    """
    Launches inference benchmark.
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch FastPitch Inference Benchmark')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_file = args.log_file
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT,
                                              args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name': 'FastPitch_PyT'})

    model = load_and_setup_model('FastPitch', parser, None, args.amp_run,
                                 'cuda', unk_args=[], forward_is_infer=True,
                                 ema=False, jitable=True)

    # FIXME Temporarily disabled due to nn.LayerNorm fp16 casting bug in pytorch:20.02-py3 and 20.03
    # model = torch.jit.script(model)

    warmup_iters = 3
    iters = 1
    gen_measures = MeasureTime()
    all_frames = 0
    for i in range(-warmup_iters, iters):
        text_padded = torch.randint(low=0, high=148, size=(args.batch_size, 128),
                                    dtype=torch.long).to('cuda')
        input_lengths = torch.IntTensor([text_padded.size(1)] * args.batch_size).to('cuda')
        durs = torch.ones_like(text_padded).mul_(4).to('cuda')

        with torch.no_grad(), gen_measures:
            mels, *_ = model(text_padded, input_lengths, dur_tgt=durs)
        num_frames = mels.size(0) * mels.size(2)

        if i >= 0:
            all_frames += num_frames
            DLLogger.log(step=(i,), data={"latency": gen_measures[-1]})
            DLLogger.log(step=(i,), data={"frames/s": num_frames / gen_measures[-1]})

    measures = gen_measures[warmup_iters:]
    DLLogger.log(step=(), data={'avg latency': np.mean(measures)})
    DLLogger.log(step=(), data={'avg frames/s': all_frames / np.sum(measures)})
    DLLogger.flush()

if __name__ == '__main__':
    main()
