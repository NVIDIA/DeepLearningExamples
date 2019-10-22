# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
import os
import subprocess
import numpy as np

from collections import OrderedDict

PERF_THR = 0.9

parser = argparse.ArgumentParser(description='Tesnorflow Benchmark Tests')

parser.add_argument('--bs', default=[1], type=int, nargs='+')
parser.add_argument('--ngpus', default=[1], type=int, nargs='+')
parser.add_argument('--perf_args', default=[], type=str, nargs='*')
parser.add_argument('--gpu_id', default=0, type=int)

parser.add_argument(
    '--mode',
    default='training',
    choices=['training', 'inference'],
    help='benchmark training or inference (default: training)'
)
parser.add_argument(
    '--bench-iterations',
    type=int,
    default=100,
    metavar='N',
    help='Run N iterations while benchmarking (ignored when training and validation)'
)
parser.add_argument(
    '--bench-warmup', type=int, default=3, metavar='N', help='Number of warmup iterations for benchmarking'
)

parser.add_argument('--precision', default='fp32', choices=['fp16', 'fp32'], help=argparse.SUPPRESS)

parser.add_argument('--baseline', type=str, default=None, metavar='FILE', help='path to the file with baselines')

parser.add_argument('--data_dir', default="/data/imagenet", type=str, metavar='<PATH>', help='path to the dataset')
parser.add_argument('--results_dir', default="/results", type=str, metavar='<PATH>', help='path to the results')

args = parser.parse_args()

command = "{{}} main.py --mode={mode}_benchmark --batch_size={batch_size} --warmup_steps={bench_warmup} --num_iter={num_iter} --precision={precision} --iter_unit=batch --data_dir={data_dir} --results_dir={results_dir}/{exp_name} --gpu_id={gpu_id} {perf_args}"

benchmark_filenames = {'training': 'training_benchmark.json', 'inference': 'eval_benchmark.json'}


def benchmark(command, metrics, args):
    sgpu = str(sys.executable)
    mgpu = "mpiexec --allow-run-as-root --bind-to socket -np {ngpu} {sgpu}"

    perf_args = []
    
    table = {k: [] for k in metrics}
    for ngpu in args.ngpus:

        row = {k: [] for k in metrics}
        for bspgpu in args.bs:

            exp_name = "{}GPU_{}BS_{}bench".format(ngpu, bspgpu, args.mode)
            results_path = os.path.join(args.results_dir, exp_name)
            os.makedirs(results_path)

            rfile = "{}/{}".format(results_path, benchmark_filenames[args.mode])
            print('rfile', rfile)
            mgpu_str = mgpu.format(ngpu=ngpu, sgpu=sgpu)
            print(mgpu_str)
            
            perf_args = [ '--' + arg for arg in args.perf_args]
            perf_args_str = " ".join(perf_args) if args.perf_args else ""
            
            cmd = command.format(
                mode=args.mode,
                batch_size=bspgpu,
                precision=args.precision,
                bench_warmup=args.bench_warmup,
                num_iter=args.bench_iterations,
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                exp_name=exp_name,
                gpu_id=args.gpu_id,
                perf_args=perf_args_str
            )
            print(cmd)

            cmd = cmd.format(sgpu if ngpu == 1 else mgpu_str)

            print(cmd.split())

            exit_code = subprocess.call(cmd.split())

            if exit_code != 0:
                print("CMD: \"{}\" exited with status {}".format("".join(cmd), exit_code))
                assert False
            else:
                print("Job ended sucessfully")

            raport = json.load(open(rfile, 'r'))

            for m in metrics:
                row[m].append((bspgpu, np.mean(raport['iter'][m])))

        for m in metrics:
            table[m].append((ngpu, OrderedDict(row[m])))

    for m in metrics:
        table[m] = OrderedDict(table[m])

    def format_float(f):
        return "{:>8.1f}".format(f)

    def format_int(i):
        return "{:>8}".format(i)

    for m in metrics:
        header = " {} |".format(m) + " |".join(map(format_int, args.ngpus)) + " |"

        print(header)
        print("-" * len(header))

        for bspgpu in args.bs:
            line = [format_int(bspgpu)]

            for ngpu in args.ngpus:
                line.append(format_float(table[m][ngpu][bspgpu]))

            print(" " * (len(m) - 7) + " |".join(line) + " |")

    return table


def load_baseline_file(path):
    with open(path, 'r') as f:
        baseline = json.load(f)
        return baseline['metrics']

    return None


def check(results, baseline, ngpus, bs, metrics):
    allright = True
    for m in metrics:
        for n in ngpus:
            for b in bs:
                result = results[m][n][b]
                try: 
                    reference = baseline[str(n)][str(b)][m]
                    if result < PERF_THR * reference:
                        allright = False
                        print(
                            "Metric: {} NGPUs: {} BS: {} Result ( {} ) is more than {} times slower than reference ( {} )".
                            format(m, n, b, result, PERF_THR, reference)
                        )
                except KeyError:
                    continue
                    
    return allright


if args.mode == 'training':
    metrics = ['total_ips']
else:
    metrics = ['total_ips']

table = benchmark(command, metrics, args)

if not args.baseline is None:
    baseline = load_baseline_file(args.baseline)

    if check(table, baseline, args.ngpus, args.bs, metrics):
        print("&&&& PASSED")
        exit(0)
    else:
        print("&&&& FAILED")
        exit(1)
