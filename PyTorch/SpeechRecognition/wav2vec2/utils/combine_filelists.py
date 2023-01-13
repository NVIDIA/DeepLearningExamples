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

from os.path import commonpath, join, relpath
import sys


def load_tsv(fpath):
    with open(fpath) as f:
        return [l.split() for l in f]


tsvs = [load_tsv(tsv) for tsv in sys.argv[1:]]
root = commonpath([t[0][0] for t in tsvs])
tsvs = [[(relpath(join(lines[0][0], p), root), frames) for p, frames in lines[1:]]
        for lines in tsvs]

print(root)
for lines in tsvs:
    for line in lines:
        print("\t".join(line))
