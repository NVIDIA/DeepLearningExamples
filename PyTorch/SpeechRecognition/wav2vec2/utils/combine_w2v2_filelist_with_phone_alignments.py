# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
from pathlib import Path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest', type=Path, nargs='+',
        help='w2v2 manifest files with <ID> <duration> on every line')
    parser.add_argument(
        '--alignments', type=Path,
        help='CPC_audio alignments with <ID> <PHONE_ID_LIST> on every line')
    parser.add_argument(
        '--ids', type=Path,
        help='List of IDs for this split (train/test, one per line)')
    parser.add_argument(
        '--out', type=Path,
        help='Output manifest fpath')

    args = parser.parse_args()

    header = None
    fpaths = {}
    durs = {}
    alis = {}
    ids = []
    out = []

    for fpath in args.manifest:
        print(f'Loading {fpath}')
        with open(fpath) as f:
            for i, line in enumerate(f):
                if i == 0:
                    header = line.strip()
                    continue
                fp, dur = line.split()
                id = Path(fp).stem
                fpaths[id] = fp
                durs[id] = dur  # int(dur)

    with open(args.alignments) as f:
        for line in f:
            id, ph = line.strip().split(' ', 1)
            alis[id] = ph

    ids = [line.strip() for line in open(args.ids)]

    for id in ids:
        fp = fpaths[id]
        d = durs[id]
        a = alis[id]
        out.append([fp, d, a])

    with open(args.out.with_suffix('.tsv'), 'w') as f:
        f.write(header + '\n')
        for o in out:
            f.write('\t'.join(o[:2]) + '\n')

    with open(args.out.with_suffix('.ph'), 'w') as f:
        for o in out:
            f.write(o[2] + '\n')
