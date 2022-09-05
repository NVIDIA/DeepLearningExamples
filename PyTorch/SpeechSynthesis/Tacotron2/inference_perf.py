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
import os
import sys
import random

from inference import checkpoint_from_distributed, unwrap_distributed, load_and_setup_model, MeasureTime, prepare_input_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-m', '--model-name', type=str, default='',
                        required=True, help='Model to train')
    parser.add_argument('--model', type=str, default='',
                        help='Full path to the model checkpoint file')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=1)
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--synth-data', action='store_true',
                        help='Test with synthetic data')
    return parser


def gen_text(use_synthetic_data):
    batch_size = 1
    text_len = 170

    if use_synthetic_data:
        text_padded = torch.randint(low=0, high=148,
                                    size=(batch_size, text_len),
                                    dtype=torch.long).cuda()
        input_lengths = torch.IntTensor([text_padded.size(1)]*
                                        batch_size).cuda().long()
    else:
        text = 'The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves. '*2
        text = [text[:text_len]]
        text_padded, input_lengths = prepare_input_sequence(text)

    return (text_padded, input_lengths)


def gen_mel(use_synthetic_data, n_mel_channels, fp16):
    if use_synthetic_data:
        batch_size = 1
        num_mels = 895
        mel_padded = torch.zeros(batch_size, n_mel_channels,
                                 num_mels).normal_(-5.62, 1.98).cuda()
    else:
        mel_padded = torch.load("data/mel.pt")

    if fp16:
        mel_padded = mel_padded.half()

    return mel_padded


def main():
    """
    Launches inference benchmark.
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_file = os.path.join(args.output, args.log_file)

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    DLLogger.metadata('infer_latency', {'unit': 's'})
    DLLogger.metadata('infer_items_per_sec', {'unit': 'items/s'})

    if args.synth_data:
        model = load_and_setup_model(args.model_name, parser, None, args.fp16,
                                     cpu_run=False, forward_is_infer=True)
    else:
        if not os.path.isfile(args.model):
            print(f"File {args.model} does not exist!")
            sys.exit(1)
        model = load_and_setup_model(args.model_name, parser, args.model,
                                     args.fp16, cpu_run=False,
                                     forward_is_infer=True)

    if args.model_name == "Tacotron2":
        model = torch.jit.script(model)

    warmup_iters = 6
    num_iters = warmup_iters + 1

    for i in range(num_iters):

        measurements = {}

        if args.model_name == 'Tacotron2':
            text_padded, input_lengths = gen_text(args.synth_data)

            with torch.no_grad(), MeasureTime(measurements, "inference_time"):
                mels, _, _ = model(text_padded, input_lengths)
            num_items = mels.size(0)*mels.size(2)

        if args.model_name == 'WaveGlow':

            n_mel_channels = model.upsample.in_channels
            mel_padded = gen_mel(args.synth_data, n_mel_channels, args.fp16)

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
