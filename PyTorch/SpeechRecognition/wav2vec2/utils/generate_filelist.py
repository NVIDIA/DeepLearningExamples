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

import soundfile
import tqdm


parser = argparse.ArgumentParser(description="Write .tsv dataset filelists")
parser.add_argument("dir", type=Path, help="Dataset directory")
parser.add_argument("output_tsv", type=Path, help="Output .tsv file path")
parser.add_argument("--extension", type=str, default="flac",
                    help="Find files with this extension")
args = parser.parse_args()

num_files = 0
print(f"Collecting .{args.extension} files in {args.dir} ...")
with open(args.output_tsv, "w") as f:
    f.write(f"{args.dir}\n")
    for fname in tqdm.tqdm(args.dir.rglob("*." + args.extension)):
        num_frames = soundfile.info(fname).frames
        f.write(f"{fname.relative_to(args.dir)}\t{num_frames}\n")
        num_files += 1
print(f"Found {num_files} files for {args.output_tsv} .")
