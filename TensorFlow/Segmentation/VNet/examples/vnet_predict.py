# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import os
import subprocess
from os.path import dirname

PARSER = argparse.ArgumentParser(description="vnet_predict")

PARSER.add_argument('--data_dir',
                    required=True,
                    type=str)

PARSER.add_argument('--model_dir',
                    required=True,
                    type=str)

PARSER.add_argument('--batch_size',
                    required=True,
                    type=int)

PARSER.add_argument('--precision',
                    choices=['fp32', 'fp16'],
                    required=True,
                    type=str)


def build_command(FLAGS, path_to_main, use_amp):
    return 'python {} --data_dir {} --model_dir {} --exec_mode predict --batch_size {} {}'.format(
        path_to_main,
        FLAGS.data_dir,
        FLAGS.model_dir,
        FLAGS.batch_size,
        use_amp)


def main():
    FLAGS = PARSER.parse_args()

    use_amp = '' if FLAGS.precision == 'fp32' else '--use_amp'
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), 'main.py')

    cmd = build_command(FLAGS, path_to_main, use_amp)

    print('Command to be executed:')
    print(cmd)
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
