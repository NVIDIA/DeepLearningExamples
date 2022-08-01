# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import pickle
import os
import socket

import torch.distributed

from fairseq import utils


def is_master(args):
    return args.distributed_rank == 0


def distributed_init(args):
    args.distributed_world_size = int(os.environ.get('WORLD_SIZE',1))
    args.distributed_rank = int(os.environ.get('RANK',0))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if args.distributed_world_size > 1:

        print('| distributed init (rank {}): env://'.format(args.distributed_rank), flush=True)
        print(f"| distributed env init. MASTER_ADDR: {os.environ['MASTER_ADDR']}, MASTER_PORT: {os.environ['MASTER_PORT']}" +
              f", WORLD_SIZE: {os.environ['WORLD_SIZE']}, RANK: {os.environ['RANK']}", flush=True)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        print("| distributed init done!", flush=True)

        suppress_output(args)
        print('| initialized host {} as rank {} and device id {}'
              .format(socket.gethostname(), args.distributed_rank, args.local_rank))


def suppress_output(main_args):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print_master(*args, **kwargs):
        if 'force' in kwargs:
            kwargs.pop('force')
        builtin_print(*args, **kwargs)

    def print(*args, **kwargs):
        if 'force' in kwargs:
            force = kwargs.pop('force')
            if force:
                builtin_print(*args, **kwargs)
    if is_master(main_args):
        __builtin__.print = print_master
    else:
        __builtin__.print = print


def all_gather_list(data, max_size=16384):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != len(all_gather_list._in_buffer):
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size)
            for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size + 2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    result = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * utils.item(out_buffer[0])) + utils.item(out_buffer[1])
        result.append(
            pickle.loads(bytes(out_buffer[2:size + 2].tolist()))
        )
    return result
