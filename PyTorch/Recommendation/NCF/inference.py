#
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


import torch.jit
import time
from argparse import ArgumentParser

import torch

from neumf import NeuMF

from logger.logger import LOGGER, timed_block, timed_function
from logger.autologging import log_hardware, log_args

from apex import amp

LOGGER.model = 'ncf'


def parse_args():
    parser = ArgumentParser(description="Benchmark inference performance of the NCF model")
    parser.add_argument('--load_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint file to be loaded before training/evaluation')
    parser.add_argument('--n_users', default=138493, type=int,
                        help='Number of users. Defaults to the number of users in the ml-20m dataset after preprocessing')
    parser.add_argument('--n_items', default=26744, type=int,
                        help='Number of items. Defaults to the number of users in the ml-20m dataset after preprocessing')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='Number of predictive factors')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability, if equal to 0 will not use dropout at all')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[256, 256, 128, 64],
                        help='Sizes of hidden layers for MLP')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--num_batches', default=20, type=int,
                        help='Number of batches for which to measure latency and throughput')
    parser.add_argument('--opt_level', default='O2', type=str,
                        help='Optimization level for Automatic Mixed Precision',
                        choices=['O0', 'O2'])

    return parser.parse_args()


def main():
    log_hardware()
    args = parse_args()
    log_args(args)

    model = NeuMF(nb_users=args.n_users, nb_items=args.n_items, mf_dim=args.factors,
                  mlp_layer_sizes=args.layers, dropout=args.dropout)

    model = model.cuda()

    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path)
        model.load_state_dict(state_dict)

    if args.opt_level == "O2":
        model = amp.initialize(model, opt_level=args.opt_level,
                               keep_batchnorm_fp32=False, loss_scale='dynamic')
    model.eval()
    
    users = torch.cuda.LongTensor(args.batch_size).random_(0, args.n_users)
    items = torch.cuda.LongTensor(args.batch_size).random_(0, args.n_items)

    latencies = []
    for _ in range(args.num_batches):
        torch.cuda.synchronize()
        start = time.time()
        predictions = model(users, items, sigmoid=True)
        torch.cuda.synchronize()
        latencies.append(time.time() - start)

    LOGGER.log(key='batch_size', value=args.batch_size)
    LOGGER.log(key='best_inference_throughput', value=args.batch_size / min(latencies))
    LOGGER.log(key='best_inference_latency', value=min(latencies))
    LOGGER.log(key='inference_latencies', value=latencies)
    return


if __name__ == '__main__':
    main()
