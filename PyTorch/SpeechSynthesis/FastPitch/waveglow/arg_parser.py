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

def parse_waveglow_args(parent, add_help=False):
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help, allow_abbrev=False)

    # misc parameters
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')

    # glow parameters
    parser.add_argument('--flows', default=12, type=int,
                        help='Number of steps of flow')
    parser.add_argument('--groups', default=8, type=int,
                        help='Number of samples in a group processed by the steps of flow')
    parser.add_argument('--early-every', default=4, type=int,
                        help='Determines how often (i.e., after how many coupling layers) \
                        a number of channels (defined by --early-size parameter) are output\
                        to the loss function')
    parser.add_argument('--early-size', default=2, type=int,
                        help='Number of channels output to the loss function')
    parser.add_argument('--sigma', default=1.0, type=float,
                        help='Standard deviation used for sampling from Gaussian')
    parser.add_argument('--segment-length', default=4000, type=int,
                        help='Segment length (audio samples) processed per iteration')

    # wavenet parameters
    wavenet = parser.add_argument_group('WaveNet parameters')
    wavenet.add_argument('--wn-kernel-size', default=3, type=int,
                        help='Kernel size for dialted convolution in the affine coupling layer (WN)')
    wavenet.add_argument('--wn-channels', default=512, type=int,
                        help='Number of channels in WN')
    wavenet.add_argument('--wn-layers', default=8, type=int,
                        help='Number of layers in WN')

    return parser
