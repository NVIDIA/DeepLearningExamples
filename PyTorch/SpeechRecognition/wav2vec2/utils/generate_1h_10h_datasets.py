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
from itertools import chain
from pathlib import Path


def load_lines(fpath):
    with open(fpath) as f:
        return [line for line in f]


parser = argparse.ArgumentParser()
parser.add_argument('ls_ft', type=Path,
                    help='Libri-light librispeech_finetuning dir')
parser.add_argument('ls_filelists', type=Path,
                    help='Directory with .tsv .wrd etc files for LibriSpeech full 960')
parser.add_argument('out', type=Path, help='Output directory')
args = parser.parse_args()

# Load LS
tsv = load_lines(args.ls_filelists / "train-full-960.tsv")
wrd = load_lines(args.ls_filelists / "train-full-960.wrd")
ltr = load_lines(args.ls_filelists / "train-full-960.ltr")

assert len(tsv) == len(wrd) + 1
assert len(ltr) == len(wrd)

files = {}
for path_frames, w, l in zip(tsv[1:], wrd, ltr):
    path, _ = path_frames.split("\t")
    key = Path(path).stem
    files[key] = (path_frames, w, l)

print(f"Loaded {len(files)} entries from {args.ls_filelists}/train-full-960")

# Load LL-LS
files_1h = list((args.ls_ft / "1h").rglob("*.flac"))
files_9h = list((args.ls_ft / "9h").rglob("*.flac"))

print(f"Found {len(files_1h)} files in the 1h dataset")
print(f"Found {len(files_9h)} files in the 9h dataset")

for name, file_iter in [("train-1h", files_1h),
                        ("train-10h", chain(files_1h, files_9h))]:

    with open(args.out / f"{name}.tsv", "w") as ftsv, \
            open(args.out / f"{name}.wrd", "w") as fwrd, \
            open(args.out / f"{name}.ltr", "w") as fltr:
        nframes = 0

        ftsv.write(tsv[0])
        for fpath in file_iter:
            key = fpath.stem
            t, w, l = files[key]
            ftsv.write(t)
            fwrd.write(w)
            fltr.write(l)
            nframes += int(t.split()[1])

        print(f"Written {nframes} frames ({nframes / 16000 / 60 / 60:.2f} h at 16kHz)")
