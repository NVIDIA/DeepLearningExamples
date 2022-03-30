# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from ast import literal_eval


def parse_hifigan_args(parent, add_help=False):
    """
    Parse model specific commandline arguments.
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help,
                                     allow_abbrev=False)
    hfg = parser.add_argument_group('HiFi-GAN generator parameters')
    hfg.add_argument('--upsample_rates', default=[8, 8, 2, 2],
                     type=literal_eval_arg,
                     help='Upsample rates')
    hfg.add_argument('--upsample_kernel_sizes', default=[16, 16, 4, 4],
                     type=literal_eval_arg,
                     help='Upsample kernel sizes')
    hfg.add_argument('--upsample_initial_channel', default=512, type=int,
                     help='Upsample initial channel')
    hfg.add_argument('--resblock', default='1', type=str,
                     help='Resblock module version')
    hfg.add_argument('--resblock_kernel_sizes', default=[3, 7, 11],
                     type=literal_eval_arg,
                     help='Resblock kernel sizes')
    hfg.add_argument('--resblock_dilation_sizes', type=literal_eval_arg,
                     default=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                     help='Resblock dilation sizes'),

    hfg = parser.add_argument_group('HiFi-GAN discriminator parameters')
    hfg.add_argument('--mpd_periods', default=[2, 3, 5, 7, 11],
                      type=literal_eval_arg,
                      help='Periods of MultiPeriodDiscriminator')
    hfg.add_argument('--concat_fwd', action='store_true',
                      help='Faster Discriminators (requires more GPU memory)')
    hfg.add_argument('--hifigan-config', type=str, default=None, required=False,
                     help='Path to a HiFi-GAN config .json'
                          ' (if provided, overrides model architecture flags)')
    return parser


def literal_eval_arg(val):
    try:
        return literal_eval(val)
    except SyntaxError as e:  # Argparse does not handle SyntaxError
        raise ValueError(str(e)) from e
