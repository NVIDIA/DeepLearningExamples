# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
""" Script that simplifies model training followed by evaluation. """

import argparse
import os
import shutil
import subprocess


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


if __name__ == '__main__':
    # CLI flags
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description=(
            'NVIDIA MaskRCNN TF2 train'
            '\n\nNote: Any additional flags not specified below will be passed to main.py'
        ),
        formatter_class=lambda prog: CustomFormatter(prog, max_help_position=100)
    )
    parser.add_argument('--gpus', type=int, metavar='N',
                        help='Number of GPU\'s. Defaults to all available')
    parser.add_argument('--batch_size', type=int, metavar='N', default=4,
                        help='Batch size used during training')
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision')
    parser.add_argument('--no_xla', action='store_true',
                        help='Disables XLA - accelerated linear algebra')
    parser.add_argument('--data_dir', type=str, metavar='DIR', default='/data',
                        help='Input directory containing the dataset')
    parser.add_argument('--weights_dir', type=str, metavar='DIR', default='/weights',
                        help='Directory containing pre-trained resnet weights')
    parser.add_argument('--no_eval', action='store_true', help='Disables evaluation after training.')

    flags, remainder = parser.parse_known_args()

    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../main.py'))
    checkpoint_path = os.path.join(flags.weights_dir, "rn50_tf_amp_ckpt_v20.06.0/nvidia_rn50_tf_amp")

    # build command
    cmd = (
        f'python {main_path}'
        f' train_and_eval'
        f' --data_dir "{flags.data_dir}"'
        f' --eval_file "{os.path.join(flags.data_dir, "annotations/instances_val2017.json")}"'
        f' --backbone_checkpoint "{checkpoint_path}"'
        f' --train_batch_size {flags.batch_size}'
    )

    if not flags.no_xla:
        cmd += ' --xla'
    if flags.amp:
        cmd += ' --amp'
    if remainder:
        cmd += ' ' + ' '.join(remainder)
    if flags.gpus is not None:
        cmd = f'CUDA_VISIBLE_DEVICES={",".join(map(str, range(flags.gpus)))} ' + cmd

    # print command
    line = '-' * shutil.get_terminal_size()[0]
    print(line, cmd, line, sep='\n', flush=True)

    # run model
    subprocess.call(cmd, shell=True)
