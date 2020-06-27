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
import numpy as np
import torch

from neumf import NeuMF

import dllogger


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
    parser.add_argument('--batch_sizes', default='1,4,16,64,256,1024,4096,16384,65536,262144,1048576', type=str,
                        help='A list of comma-separated batch size values to benchmark')
    parser.add_argument('--num_batches', default=200, type=int,
                        help='Number of batches for which to measure latency and throughput')
    parser.add_argument('--fp16', action='store_true', help='Cast the model to FP16 precision', default=False)
    parser.add_argument('--log_path', default='log.json', type=str,
                        help='Path for the JSON training log')

    return parser.parse_args()


def main():
    args = parse_args()
    dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                       filename=args.log_path),
                            dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])

    dllogger.log(data=vars(args), step='PARAMETER')

    model = NeuMF(nb_users=args.n_users, nb_items=args.n_items, mf_dim=args.factors,
                  mlp_layer_sizes=args.layers, dropout=args.dropout)

    model = model.cuda()

    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path)
        model.load_state_dict(state_dict)

    if args.fp16:
        model.half()
    model.eval()
    
    batch_sizes = args.batch_sizes.split(',')
    batch_sizes = [int(s) for s in batch_sizes]

    result_data = {}
    for batch_size in batch_sizes:
        print('benchmarking batch size: ', batch_size)
        users = torch.cuda.LongTensor(batch_size).random_(0, args.n_users)
        items = torch.cuda.LongTensor(batch_size).random_(0, args.n_items)

        latencies = []
        for _ in range(args.num_batches):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(users, items, sigmoid=True)
            torch.cuda.synchronize()
            latencies.append(time.time() - start)

        result_data[f'batch_{batch_size}_mean_throughput'] = batch_size / np.mean(latencies)
        result_data[f'batch_{batch_size}_mean_latency'] = np.mean(latencies)
        result_data[f'batch_{batch_size}_p90_latency'] = np.percentile(latencies, 0.90)
        result_data[f'batch_{batch_size}_p95_latency'] = np.percentile(latencies, 0.95)
        result_data[f'batch_{batch_size}_p99_latency'] = np.percentile(latencies, 0.99)

    dllogger.log(data=result_data, step=tuple())
    dllogger.flush()
    return


if __name__ == '__main__':
    main()
