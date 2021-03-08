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


import os
import argparse


def parse_args(parser):
    """
        Parse commandline arguments.
    """
    parser.add_argument("--trtis_model_name",
                        type=str,
                        default='waveglow',
                        help="exports to appropriate directory for TRTIS")
    parser.add_argument("--trtis_model_version",
                        type=int,
                        default=1,
                        help="exports to appropriate directory for TRTIS")
    parser.add_argument('--fp16', action='store_true',
                        help='inference with mixed precision')
    return parser


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch WaveGlow TRTIS config exporter')
    parser = parse_args(parser)
    args = parser.parse_args()

    # prepare repository
    model_folder = os.path.join('./trtis_repo', args.trtis_model_name)
    version_folder = os.path.join(model_folder, str(args.trtis_model_version))
    if not os.path.exists(version_folder):
        os.makedirs(version_folder)

    # build the config for TRTIS
    config_filename = os.path.join(model_folder, "config.pbtxt")
    config_template = r"""
name: "{model_name}"
platform: "tensorrt_plan"
default_model_filename: "waveglow_fp16.engine"

max_batch_size: 1

input {{
  name: "mel"
  data_type: {fp_type}
  dims: [80, -1, 1]
}}
input {{
  name: "z"
  data_type: {fp_type}
  dims: [8, -1, 1]
}}
output {{
  name: "audio"
  data_type: {fp_type}
  dims: [-1]
}}
"""

    config_values = {
        "model_name": args.trtis_model_name,
        "fp_type": "TYPE_FP16" if args.fp16 else "TYPE_FP32"
    }

    with open(model_folder + "/config.pbtxt", "w") as file:
        final_config_str = config_template.format_map(config_values)
        file.write(final_config_str)


if __name__ == '__main__':
    main()
