#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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



import os
import socket
import subprocess

from train import main as single_process_main
from fairseq import distributed_utils, options


def main(args):
    if args.distributed_init_method is None and args.distributed_port > 0:
        # We can determine the init method automatically for Slurm.
        node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port)
                args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError as e:  # Slurm is not installed
                pass
    if args.distributed_init_method is None:
        raise ValueError('--distributed-init-method or --distributed-port '
                         'must be specified for distributed training')

    args.distributed_rank = distributed_utils.distributed_init(args)
    args.device_id = int(os.environ.get('LOCAL_RANK', args.local_rank))
    print('| initialized host {} as rank {} and device id {}'.format(socket.gethostname(), args.distributed_rank, args.device_id))
    single_process_main(args)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
