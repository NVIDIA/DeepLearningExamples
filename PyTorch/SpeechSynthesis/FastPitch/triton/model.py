# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)+'/../'))

from common.text import symbols
from inference import load_model_from_ckpt
import models
from torch.utils.data import DataLoader
import torch
import numpy as np

def update_argparser(parser):

    ### copy-paste from ./fastpitch/arg_parser.py
    io = parser.add_argument_group('io parameters')
    io.add_argument('--n-mel-channels', default=80, type=int,
                    help='Number of bins in mel-spectrograms')

    symbols = parser.add_argument_group('symbols parameters')
    symbols.add_argument('--n-symbols', default=148, type=int,
                         help='Number of symbols in dictionary')
    symbols.add_argument('--padding-idx', default=0, type=int,
                         help='Index of padding symbol in dictionary')
    symbols.add_argument('--symbols-embedding-dim', default=384, type=int,
                         help='Input embedding dimension')

    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--symbol-set', type=str, default='english_basic',
                                 help='Define symbol set for input text')

    in_fft = parser.add_argument_group('input FFT parameters')
    in_fft.add_argument('--in-fft-n-layers', default=6, type=int,
                        help='Number of FFT blocks')
    in_fft.add_argument('--in-fft-n-heads', default=1, type=int,
                        help='Number of attention heads')
    in_fft.add_argument('--in-fft-d-head', default=64, type=int,
                        help='Dim of attention heads')
    in_fft.add_argument('--in-fft-conv1d-kernel-size', default=3, type=int,
                        help='Conv-1D kernel size')
    in_fft.add_argument('--in-fft-conv1d-filter-size', default=1536, type=int,
                        help='Conv-1D filter size')
    in_fft.add_argument('--in-fft-output-size', default=384, type=int,
                        help='Output dim')
    in_fft.add_argument('--p-in-fft-dropout', default=0.1, type=float,
                        help='Dropout probability')
    in_fft.add_argument('--p-in-fft-dropatt', default=0.1, type=float,
                        help='Multi-head attention dropout')
    in_fft.add_argument('--p-in-fft-dropemb', default=0.0, type=float,
                        help='Dropout added to word+positional embeddings')

    out_fft = parser.add_argument_group('output FFT parameters')
    out_fft.add_argument('--out-fft-n-layers', default=6, type=int,
                         help='Number of FFT blocks')
    out_fft.add_argument('--out-fft-n-heads', default=1, type=int,
                         help='Number of attention heads')
    out_fft.add_argument('--out-fft-d-head', default=64, type=int,
                         help='Dim of attention head')
    out_fft.add_argument('--out-fft-conv1d-kernel-size', default=3, type=int,
                         help='Conv-1D kernel size')
    out_fft.add_argument('--out-fft-conv1d-filter-size', default=1536, type=int,
                         help='Conv-1D filter size')
    out_fft.add_argument('--out-fft-output-size', default=384, type=int,
                         help='Output dim')
    out_fft.add_argument('--p-out-fft-dropout', default=0.1, type=float,
                         help='Dropout probability for out_fft')
    out_fft.add_argument('--p-out-fft-dropatt', default=0.1, type=float,
                         help='Multi-head attention dropout')
    out_fft.add_argument('--p-out-fft-dropemb', default=0.0, type=float,
                         help='Dropout added to word+positional embeddings')

    dur_pred = parser.add_argument_group('duration predictor parameters')
    dur_pred.add_argument('--dur-predictor-kernel-size', default=3, type=int,
                          help='Duration predictor conv-1D kernel size')
    dur_pred.add_argument('--dur-predictor-filter-size', default=256, type=int,
                          help='Duration predictor conv-1D filter size')
    dur_pred.add_argument('--p-dur-predictor-dropout', default=0.1, type=float,
                          help='Dropout probability for duration predictor')
    dur_pred.add_argument('--dur-predictor-n-layers', default=2, type=int,
                          help='Number of conv-1D layers')

    pitch_pred = parser.add_argument_group('pitch predictor parameters')
    pitch_pred.add_argument('--pitch-predictor-kernel-size', default=3, type=int,
                            help='Pitch predictor conv-1D kernel size')
    pitch_pred.add_argument('--pitch-predictor-filter-size', default=256, type=int,
                            help='Pitch predictor conv-1D filter size')
    pitch_pred.add_argument('--p-pitch-predictor-dropout', default=0.1, type=float,
                            help='Pitch probability for pitch predictor')
    pitch_pred.add_argument('--pitch-predictor-n-layers', default=2, type=int,
                            help='Number of conv-1D layers')

    energy_pred = parser.add_argument_group('energy predictor parameters')
    energy_pred.add_argument('--energy-conditioning', type=bool, default=True)
    energy_pred.add_argument('--energy-predictor-kernel-size', default=3, type=int,
                            help='Pitch predictor conv-1D kernel size')
    energy_pred.add_argument('--energy-predictor-filter-size', default=256, type=int,
                            help='Pitch predictor conv-1D filter size')
    energy_pred.add_argument('--p-energy-predictor-dropout', default=0.1, type=float,
                            help='Pitch probability for energy predictor')
    energy_pred.add_argument('--energy-predictor-n-layers', default=2, type=int,
                            help='Number of conv-1D layers')

    ###~copy-paste from ./fastpitch/arg_parser.py

    parser.add_argument('--checkpoint', type=str,
                        help='Full path to the FastPitch checkpoint file')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model \
(if saved in checkpoints)')

    cond = parser.add_argument_group('conditioning parameters')
    cond.add_argument('--pitch-embedding-kernel-size', default=3, type=int,
                      help='Pitch embedding conv-1D kernel size')
    cond.add_argument('--energy-embedding-kernel-size', default=3, type=int,
                      help='Pitch embedding conv-1D kernel size')
    cond.add_argument('--speaker-emb-weight', type=float, default=1.0,
                      help='Scale speaker embedding')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Number of speakers in the model.')
    cond.add_argument('--pitch-conditioning-formants', default=1, type=int,
                      help='Number of speech formants to condition on.')
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16"],
                        help="PyTorch model precision")
    parser.add_argument("--output-format", type=str, required=True,
                        help="Output format")


def get_model(**model_args):

    import argparse
    args = argparse.Namespace(**model_args)

    model_config = models.get_model_config(model_name="FastPitch",
                                           args=args)

    jittable = True if 'ts-' in args.output_format else False

    model = models.get_model(model_name="FastPitch",
                             model_config=model_config,
                             device='cuda',
                             forward_is_infer=True,
                             jitable=jittable)
    model = load_model_from_ckpt(args.checkpoint, args.ema, model)
    if args.precision == "fp16":
        model = model.half()
    model.eval()
    tensor_names = {"inputs": ["INPUT__0"],
                    "outputs" : ["OUTPUT__0", "OUTPUT__1",
                                 "OUTPUT__2", "OUTPUT__3", "OUTPUT__4"]}

    return model, tensor_names
