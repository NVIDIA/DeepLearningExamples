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

import models
import torch
import argparse
import numpy as np
import json
import time

from inference import checkpoint_from_distributed, unwrap_distributed, load_and_setup_model, MeasureTime

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from apex import amp

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-m', '--model-name', type=str, default='', required=True,
                        help='Model to train')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
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
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_file = args.log_file

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT,
                                              args.output+'/'+args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    model = load_and_setup_model(args.model_name, parser, None, args.amp_run,
                                 forward_is_infer=True)

    if args.model_name == "Tacotron2":
        model = torch.jit.script(model)

    warmup_iters = 3
    num_iters = 1+warmup_iters

    for i in range(num_iters):

        measurements = {}

        if args.model_name == 'Tacotron2':
            text_padded = torch.randint(low=0, high=148, size=(args.batch_size, 140),
                                        dtype=torch.long).cuda()
            input_lengths = torch.IntTensor([text_padded.size(1)]*args.batch_size).cuda().long()
            with torch.no_grad(), MeasureTime(measurements, "inference_time"):
                mels, _, _ = model(text_padded, input_lengths)
            num_items = mels.size(0)*mels.size(2)

        if args.model_name == 'WaveGlow':
            n_mel_channels = model.upsample.in_channels
            num_mels = 895
            mel_padded = torch.zeros(args.batch_size, n_mel_channels,
                                     num_mels).normal_(-5.62, 1.98).cuda()
            if args.amp_run:
                mel_padded = mel_padded.half()

            with torch.no_grad(), MeasureTime(measurements, "inference_time"):
                audios = model(mel_padded)
                audios = audios.float()
            num_items = audios.size(0)*audios.size(1)

        if i >= warmup_iters:
            DLLogger.log(step=(i-warmup_iters,), data={"latency": measurements['inference_time']})
            DLLogger.log(step=(i-warmup_iters,), data={"items_per_sec": num_items/measurements['inference_time']})

    DLLogger.log(step=tuple(),
                 data={'infer_latency': measurements['inference_time']})
    DLLogger.log(step=tuple(),
                 data={'infer_items_per_sec': num_items/measurements['inference_time']})

    DLLogger.flush()

if __name__ == '__main__':
    main()
