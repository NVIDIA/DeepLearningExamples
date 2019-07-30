# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
import sys
from subprocess import Popen, PIPE

parser = argparse.ArgumentParser(description='Translate')
parser.add_argument('--executable', default='nmt.py', help='path to nmt.py')
parser.add_argument('--infer_batch_size', metavar='B1,[B2,...]',
                    default='64', help='batch sizes separated by comma')
parser.add_argument('--beam_width', metavar='W1,[W2,...]',
                    default='5', help='beam widths separated by comma')
args, other_args = parser.parse_known_args()

batch_sizes = list(map(int, args.infer_batch_size.split(',')))
beam_widths = list(map(int, args.beam_width.split(',')))


def pr(*args, column_len=14):
  for arg in args:
    if type(arg) is float:
      arg = '{:.2f}'.format(arg)
    arg = str(arg)

    print('', arg.ljust(column_len), end=' |')
  print()

pr('batch size', 'beam width', 'bleu', 'sentences/sec', 'tokens/sec',
   'latency_avg', 'latency_50', 'latency_90', 'latency_95', 'latency_99', 'latency_100')

for batch_size in batch_sizes:
  for beam_width in beam_widths:
    cmd = ['python', args.executable, '--beam_width', str(beam_width),
           '--infer_batch_size', str(batch_size), '--mode', 'infer'] + other_args
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()

    bleu_search_res = re.search(rb'\nbleu is ((\d|.)+)', out)
    speed_search_res = re.search(
      rb'\neval time for ckpt: ((\d|.)+) mins \(((\d|.)+) sent/sec, ((\d|.)+) tokens/sec\)', out)

    latencies = []
    for lat in ['avg', '50', '90', '95', '99', '100']:
      latencies.append(re.search(r'\neval latency_{} for ckpt: ((\d|.)+) ms'.format(lat).encode(), out))
    if bleu_search_res is None or speed_search_res is None or any(filter(lambda x: x is None, latencies)):
      print('AN ERROR OCCURRED WHILE RUNNING:', cmd, file=sys.stderr)
      print('-' * 20, 'STDOUT', '-' * 20, file=sys.stderr)
      print(out.decode())
      print('-' * 20, 'STDERR', '-' * 20, file=sys.stderr)
      print(err.decode())
      exit(1)

    bleu = float(bleu_search_res.group(1))
    sentences_per_sec, tokens_per_sec = map(float, speed_search_res.group(3, 5))
    latencies = list(map(lambda x: float(x.group(1)), latencies))

    pr(batch_size, beam_width, bleu, sentences_per_sec, tokens_per_sec, *latencies)
