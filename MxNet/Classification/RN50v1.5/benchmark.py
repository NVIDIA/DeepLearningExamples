#!/usr/bin/env python3

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
import json
import sys
import tempfile
import json
import os
import traceback
import numpy as np
from collections import OrderedDict
from subprocess import Popen

def int_list(x):
    return list(map(int, x.split(',')))

parser = argparse.ArgumentParser(description='Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--executable', default='./runner', help='path to runner')
parser.add_argument('-o', '--output', metavar='OUT', required=True, help="path to benchmark report")
parser.add_argument('-n', '--ngpus', metavar='N1,[N2,...]', type=int_list,
                    required=True, help='numbers of gpus separated by comma')
parser.add_argument('-b', '--batch-sizes', metavar='B1,[B2,...]', type=int_list,
                    required=True, help='batch sizes separated by comma')
parser.add_argument('-i', '--benchmark-iters', metavar='I',
                    type=int, default=100, help='iterations')
parser.add_argument('-e', '--epochs', metavar='E',
                    type=int, default=1, help='number of epochs')
parser.add_argument('-w', '--warmup', metavar='N',
                    type=int, default=0, help='warmup epochs')
parser.add_argument('--timeout', metavar='T',
                    type=str, default='inf', help='timeout for each run')
parser.add_argument('--mode', metavar='MODE', choices=('train_val', 'train', 'val'), default='train_val',
                    help="benchmark mode")
args, other_args = parser.parse_known_args()

latency_percentiles = ['avg', 50, 90, 95, 99, 100]
harmonic_mean_metrics = ['train.total_ips', 'val.total_ips']

res = OrderedDict()
res['model'] = ''
res['ngpus'] = args.ngpus
res['bs'] = args.batch_sizes
res['metric_keys'] = []
if args.mode == 'train' or args.mode == 'train_val':
    res['metric_keys'].append('train.total_ips')
    for percentile in latency_percentiles:
        res['metric_keys'].append('train.latency_{}'.format(percentile))
if args.mode == 'val' or args.mode == 'train_val':
    res['metric_keys'].append('val.total_ips')
    for percentile in latency_percentiles:
        res['metric_keys'].append('val.latency_{}'.format(percentile))

res['metrics'] = OrderedDict()

for n in args.ngpus:
    res['metrics'][str(n)] = OrderedDict()
    for bs in args.batch_sizes:
        res['metrics'][str(n)][str(bs)] = OrderedDict()

        report_file = args.output + '-{},{}'.format(n, bs)
        Popen(['timeout', args.timeout, args.executable, '-n', str(n), '-b', str(bs),
               '--benchmark-iters', str(args.benchmark_iters),
               '-e', str(args.epochs), '--report', report_file,
               '--mode', args.mode, '--no-metrics'] + other_args,
              stdout=sys.stderr).wait()

        try:
            for suffix in ['', *['-{}'.format(i) for i in range(1, n)]]:
                try:
                    with open(report_file + suffix, 'r') as f:
                        report = json.load(f)
                    break
                except FileNotFoundError:
                    pass
            else:
                with open(report_file, 'r') as f:
                    report = json.load(f)

            for metric in res['metric_keys']:
                if len(report['metrics'][metric]) != args.epochs:
                    raise ValueError('Wrong number epochs in report')
                data = report['metrics'][metric][args.warmup:]
                if metric in harmonic_mean_metrics:
                    avg = len(data) / sum(map(lambda x: 1 / x, data))
                else:
                    avg = np.mean(data)
                res['metrics'][str(n)][str(bs)][metric] = avg
        except Exception as e:
            traceback.print_exc()

            for metric in res['metric_keys']:
                res['metrics'][str(n)][str(bs)][metric] = float('nan')


column_len = 11
for m in res['metric_keys']:
    print(m, file=sys.stderr)
    print(' ' * column_len, end='|', file=sys.stderr)
    for bs in args.batch_sizes:
        print(str(bs).center(column_len), end='|', file=sys.stderr)
    print(file=sys.stderr)
    print('-' * (len(args.batch_sizes) + 1) * (column_len + 1), file=sys.stderr)
    for n in args.ngpus:
        print(str(n).center(column_len), end='|', file=sys.stderr)
        for bs in args.batch_sizes:
            print('{:.5g}'.format(res['metrics'][str(n)][str(bs)][m]).center(column_len), end='|', file=sys.stderr)
        print(file=sys.stderr)
    print(file=sys.stderr)


with open(args.output, 'w') as f:
    json.dump(res, f, indent=4)
