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
from collections import OrderedDict
from subprocess import Popen

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('--executable', default='./runner', help='path to runner')
parser.add_argument('-n', '--ngpus', metavar='N1,[N2,...]',
                    required=True, help='numbers of gpus separated by comma')
parser.add_argument('-b', '--batch-sizes', metavar='B1,[B2,...]',
                    required=True, help='batch sizes separated by comma')
parser.add_argument('-i', '--benchmark-iters', metavar='I',
                    type=int, default=100, help='iterations')
parser.add_argument('-e', '--epochs', metavar='E',
                    type=int, default=1, help='number of epochs')
parser.add_argument('-w', '--warmup', metavar='N',
                    type=int, default=0, help='warmup epochs')
parser.add_argument('-o', '--output', metavar='OUT', required=True, help="path to benchmark report")
parser.add_argument('--only-inference', action='store_true', help="benchmark inference only")
args, other_args = parser.parse_known_args()

ngpus = list(map(int, args.ngpus.split(',')))
batch_sizes = list(map(int, args.batch_sizes.split(',')))


res = OrderedDict()
res['model'] = ''
res['ngpus'] = ngpus
res['bs'] = batch_sizes
if args.only_inference:
    res['metric_keys'] = ['val.total_ips']
else:
    res['metric_keys'] = ['train.total_ips', 'val.total_ips']
res['metrics'] = OrderedDict()

for n in ngpus:
    res['metrics'][str(n)] = OrderedDict()
    for bs in batch_sizes:
        res['metrics'][str(n)][str(bs)] = OrderedDict()

        report_file = args.output + '-{},{}'.format(n, bs)
        Popen([args.executable, '-n', str(n), '-b', str(bs),
               '--benchmark-iters', str(args.benchmark_iters),
               '-e', str(args.epochs), '--report', report_file,
               *([] if not args.only_inference else ['--only-inference']),
               '--no-metrics'] + other_args, stdout=sys.stderr).wait()

        with open(report_file, 'r') as f:
            report = json.load(f)

        for metric in res['metric_keys']:
            data = report['metrics'][metric][args.warmup:]
            avg = len(data) / sum(map(lambda x: 1 / x, data))
            res['metrics'][str(n)][str(bs)][metric] = avg


column_len = 7
for m in res['metric_keys']:
    print(m, file=sys.stderr)
    print(' ' * column_len, end='|', file=sys.stderr)
    for bs in batch_sizes:
        print(str(bs).center(column_len), end='|', file=sys.stderr)
    print(file=sys.stderr)
    print('-' * (len(batch_sizes) + 1) * (column_len + 1), file=sys.stderr)
    for n in ngpus:
        print(str(n).center(column_len), end='|', file=sys.stderr)
        for bs in batch_sizes:
            print(str(round(res['metrics'][str(n)][str(bs)][m])).center(column_len), end='|', file=sys.stderr)
        print(file=sys.stderr)
    print(file=sys.stderr)


with open(args.output, 'w') as f:
    json.dump(res, f, indent=4)
