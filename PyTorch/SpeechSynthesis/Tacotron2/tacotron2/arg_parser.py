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

from tacotron2.text import symbols


def tacotron2_parser(parent, add_help=False):
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)

    # misc parameters
    parser.add_argument('--mask-padding', default=False, type=bool,
                        help='Use mask padding')
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')

    # symbols parameters
    global symbols
    len_symbols = len(symbols)
    symbols = parser.add_argument_group('symbols parameters')
    symbols.add_argument('--n-symbols', default=len_symbols, type=int,
                         help='Number of symbols in dictionary')
    symbols.add_argument('--symbols-embedding-dim', default=512, type=int,
                         help='Input embedding dimension')

    # encoder parameters
    encoder = parser.add_argument_group('encoder parameters')
    encoder.add_argument('--encoder-kernel-size', default=5, type=int,
                         help='Encoder kernel size')
    encoder.add_argument('--encoder-n-convolutions', default=3, type=int,
                         help='Number of encoder convolutions')
    encoder.add_argument('--encoder-embedding-dim', default=512, type=int,
                         help='Encoder embedding dimension')

    # decoder parameters
    decoder = parser.add_argument_group('decoder parameters')
    decoder.add_argument('--n-frames-per-step', default=1,
                         type=int,
                         help='Number of frames processed per step') # currently only 1 is supported
    decoder.add_argument('--decoder-rnn-dim', default=1024, type=int,
                         help='Number of units in decoder LSTM')
    decoder.add_argument('--prenet-dim', default=256, type=int,
                         help='Number of ReLU units in prenet layers')
    decoder.add_argument('--max-decoder-steps', default=2000, type=int,
                         help='Maximum number of output mel spectrograms')
    decoder.add_argument('--gate-threshold', default=0.5, type=float,
                         help='Probability threshold for stop token')
    decoder.add_argument('--p-attention-dropout', default=0.1, type=float,
                         help='Dropout probability for attention LSTM')
    decoder.add_argument('--p-decoder-dropout', default=0.1, type=float,
                         help='Dropout probability for decoder LSTM')
    decoder.add_argument('--decoder-no-early-stopping', action='store_true',
                         help='Stop decoding once all samples are finished')

    # attention parameters
    attention = parser.add_argument_group('attention parameters')
    attention.add_argument('--attention-rnn-dim', default=1024, type=int,
                           help='Number of units in attention LSTM')
    attention.add_argument('--attention-dim', default=128, type=int,
                           help='Dimension of attention hidden representation')

    # location layer parameters
    location = parser.add_argument_group('location parameters')
    location.add_argument(
        '--attention-location-n-filters', default=32, type=int,
        help='Number of filters for location-sensitive attention')
    location.add_argument(
        '--attention-location-kernel-size', default=31, type=int,
        help='Kernel size for location-sensitive attention')

    # Mel-post processing network parameters
    postnet = parser.add_argument_group('postnet parameters')
    postnet.add_argument('--postnet-embedding-dim', default=512, type=int,
                         help='Postnet embedding dimension')
    postnet.add_argument('--postnet-kernel-size', default=5, type=int,
                         help='Postnet kernel size')
    postnet.add_argument('--postnet-n-convolutions', default=5, type=int,
                         help='Number of postnet convolutions')

    return parser
