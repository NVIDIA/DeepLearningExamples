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

PARSER = argparse.ArgumentParser(description="vnet_benchmark")

PARSER.add_argument('--data_dir',
                    required=True,
                    type=str)

PARSER.add_argument('--model_dir',
                    required=True,
                    type=str)

PARSER.add_argument('--mode',
                    choices=['train', 'predict'],
                    required=True,
                    type=str)

PARSER.add_argument('--gpus',
                    choices=[1, 8],
                    required=True,
                    type=int)

PARSER.add_argument('--batch_size',
                    required=True,
                    type=int)

PARSER.add_argument('--precision',
                    choices=['fp32', 'fp16'],
                    required=True,
                    type=str)


def build_horovod_prefix(gpus):
    return 'mpirun -np {} -H localhost:{} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca ' \
           'pml ob1 -mca btl ^openib --allow-run-as-root '.format(gpus, gpus)


def build_command(FLAGS, path_to_main, use_amp):
    return 'python {} --data_dir {} --model_dir {} --exec_mode {} --batch_size {} {} --augment --benchmark'.format(
        path_to_main,
        FLAGS.data_dir,
        FLAGS.model_dir,
        FLAGS.mode,
        FLAGS.batch_size,
        use_amp)


def main():
    FLAGS = PARSER.parse_args()

    use_amp = '--use_amp' if FLAGS.precision == 'fp16' else ''
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), 'main.py')

    cmd = build_command(FLAGS, path_to_main, use_amp)

    if FLAGS.gpus > 1:
        assert FLAGS.mode != 'predict', 'Prediction can only be benchmarked on 1 GPU'
        cmd = build_horovod_prefix(FLAGS.gpus) + cmd

    print('Command to be executed:')
    print(cmd)
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
