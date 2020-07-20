# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str)
parser.add_argument('outfile', type=str)
args = parser.parse_args()

with open(args.infile, 'r') as infile:
    with open(args.outfile, 'w') as outfile:
        for line in infile.readlines():
            line = line.strip().split()
            if line[-1] == '</s>':
                line.pop()
            if line[0][0] == '▁':
                s = line[0][1:]
            else:
                s = line[0]
            for w in line[1:]:
                if w[0] == '▁':
                    s += ' ' + w[1:]
                else:
                    s += w
            s += '\n'
            outfile.write(s)
