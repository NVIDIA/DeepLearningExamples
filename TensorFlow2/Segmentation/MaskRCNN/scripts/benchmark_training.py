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
""" Script that simplifies running training benchmark """

import argparse
import os
import shutil
import subprocess
from pathlib import Path

LOCK_FILE = Path('/tmp/mrcnn_tf2.lock')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


if __name__ == '__main__':
    # CLI flags
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description=(
            'NVIDIA MaskRCNN TF2 train benchmark'
            '\n\nNote: Any additional flags not specified below will be passed to main.py'
        ),
        formatter_class=lambda prog: CustomFormatter(prog, max_help_position=100)
    )
    parser.add_argument('--gpus', type=int, metavar='N',
                        help='Number of GPU\'s. Defaults to all available')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size used during training')
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision')
    parser.add_argument('--no_xla', action='store_true',
                        help='Disables XLA - accelerated linear algebra')
    parser.add_argument('--data_dir', type=str, metavar='DIR', default='/data',
                        help='Input directory containing the dataset')
    parser.add_argument('--weights_dir', type=str, metavar='DIR', default='/weights',
                        help='Directory containing pre-trained resnet weights')
    parser.add_argument('--slurm_lock', action='store_true',
                        help='Prevent this script from being launched multiple times when used in multi-gpu slurm setup')

    flags, remainder = parser.parse_known_args()

    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../main.py'))
    checkpoint_path = os.path.join(flags.weights_dir, "rn50_tf_amp_ckpt_v20.06.0/nvidia_rn50_tf_amp")

    # build command
    cmd = (
        f'python {main_path}'
        f' train'
        f' --data_dir "{flags.data_dir}"'
        f' --backbone_checkpoint "{checkpoint_path}"'
        f' --epochs 1'
        f' --steps_per_epoch 200'
        f' --log_every 10'
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

    # acquire lock if --slurm_lock is provided
    try:
        flags.slurm_lock and LOCK_FILE.touch(exist_ok=False)
    except FileExistsError:
        print(f'Failed to acquire lock ({LOCK_FILE}) - skipping')
        exit(0)

    # run model
    code = subprocess.call(cmd, shell=True)

    flags.slurm_lock and LOCK_FILE.unlink()
    exit(code)
