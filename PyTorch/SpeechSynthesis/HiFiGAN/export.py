# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import torch

import models


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('model_name', type=str,
                        choices=['HiFi-GAN', 'FastPitch'],
                        help='Name of the converted model')
    parser.add_argument('input_ckpt', type=str,
                        help='Path to the input checkpoint')
    parser.add_argument('output_ckpt', default=None,
                        help='Path to save the output checkpoint to')
    parser.add_argument('--cuda', action='store_true',
                        help='Move model weights to GPU before export')
    parser.add_argument('--amp', action='store_true',
                        help='Convert model to FP16 prior to saving')
    parser.add_argument('--load-from', type=str, default='pyt',
                        choices=['pyt', 'ts'],
                        help='Source checkpoint format')
    parser.add_argument('--convert-to', type=str, default='ts',
                        choices=['ts', 'ttrt'],
                        help='Output checkpoint format')
    return parser


def main():
    """
    Exports PyT or TorchScript checkpoint to TorchScript or Torch-TensorRT.
    """
    parser = argparse.ArgumentParser(description='PyTorch model export',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    device = torch.device('cuda' if args.cuda else 'cpu')

    assert args.load_from != args.convert_to, \
        'Load and convert formats must be different'

    print(f'Converting {args.model_name} from "{args.load_from}"'
          f' to "{args.convert_to}" ({device}).')

    if args.load_from == 'ts':
        ts_model, _ = models.load_and_setup_ts_model(args.model_name,
                                                     args.input_ckpt, args.amp,
                                                     device)
    else:
        assert args.load_from == 'pyt'
        pyt_model, _ = models.load_pyt_model_for_infer(
            args.model_name, parser, args.input_ckpt, args.amp, device,
            unk_args=unk_args, jitable=True)
        ts_model = torch.jit.script(pyt_model)

    if args.convert_to == 'ts':
        torch.jit.save(ts_model, args.output_ckpt)
    else:
        assert args.convert_to == 'ttrt'

        trt_model = models.convert_ts_to_trt('HiFi-GAN', ts_model, parser,
                                             args.amp, unk_args)
        torch.jit.save(trt_model, args.output_ckpt)

    print(f'{args.model_name}: checkpoint saved to {args.output_ckpt}.')

    if unk_args:
        print(f'Warning: encountered unknown program options: {unk_args}')


if __name__ == '__main__':
    main()
