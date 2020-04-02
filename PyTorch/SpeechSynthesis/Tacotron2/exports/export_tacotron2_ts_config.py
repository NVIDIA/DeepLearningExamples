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
                        default='tacotron2',
                        help="exports to appropriate directory for TRTIS")
    parser.add_argument("--trtis_model_version",
                        type=int,
                        default=1,
                        help="exports to appropriate directory for TRTIS")
    parser.add_argument("--trtis_max_batch_size",
                        type=int,
                        default=8,
                        help="Specifies the 'max_batch_size' in the TRTIS model config.\
                              See the TRTIS documentation for more info.")
    parser.add_argument('--amp-run', action='store_true',
                        help='inference with AMP')
    return parser


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 TRTIS config exporter')
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
platform: "pytorch_libtorch"
max_batch_size: {max_batch_size}
input [
  {{
    name: "sequence__0"
    data_type: TYPE_INT64
    dims: [-1]
  }},
  {{
    name: "input_lengths__1"
    data_type: TYPE_INT64
    dims: [1]
    reshape: {{ shape: [ ] }}
  }}
]
output [
  {{
    name: "mel_outputs_postnet__0"
    data_type: {fp_type}
    dims: [80,-1]
  }},
  {{
    name: "mel_lengths__1"
    data_type: TYPE_INT32
    dims: [1]
    reshape: {{ shape: [ ] }}
  }},
  {{
    name: "alignments__2"
    data_type: {fp_type}
    dims: [-1,-1]
  }}
]
"""
    
    config_values = {
        "model_name": args.trtis_model_name,
        "max_batch_size": args.trtis_max_batch_size,
        "fp_type": "TYPE_FP16" if args.amp_run else "TYPE_FP32"
    }
    
    with open(model_folder + "/config.pbtxt", "w") as file:
        final_config_str = config_template.format_map(config_values)
        file.write(final_config_str)


if __name__ == '__main__':
    main()

