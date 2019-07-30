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


#!/usr/bin/env python
import argparse
import os
import glob
import multiprocessing
import json

import pandas as pd

from preprocessing_utils import parallel_preprocess

parser = argparse.ArgumentParser(description='Preprocess LibriSpeech.')
parser.add_argument('--input_dir', type=str, required=True,
                    help='LibriSpeech collection input dir')
parser.add_argument('--dest_dir', type=str, required=True,
                    help='Output dir')
parser.add_argument('--output_json', type=str, default='./',
                    help='name of the output json file.')
parser.add_argument('-s','--speed', type=float, nargs='*',
                    help='Speed perturbation ratio')
parser.add_argument('--target_sr', type=int, default=None,
                    help='Target sample rate. '
                         'defaults to the input sample rate')
parser.add_argument('--overwrite', action='store_true',
                    help='Overwrite file if exists')
parser.add_argument('--parallel', type=int, default=multiprocessing.cpu_count(),
                    help='Number of threads to use when processing audio files')
args = parser.parse_args()

args.input_dir = args.input_dir.rstrip('/')
args.dest_dir = args.dest_dir.rstrip('/')

def build_input_arr(input_dir):
    txt_files = glob.glob(os.path.join(input_dir, '**', '*.trans.txt'),
                          recursive=True)
    input_data = []
    for txt_file in txt_files:
        rel_path = os.path.relpath(txt_file, input_dir)
        with open(txt_file) as fp:
            for line in fp:
                fname, _, transcript = line.partition(' ')
                input_data.append(dict(input_relpath=os.path.dirname(rel_path),
                                       input_fname=fname+'.flac',
                                       transcript=transcript))
    return input_data


print("[%s] Scaning input dir..." % args.output_json)
dataset = build_input_arr(input_dir=args.input_dir)

print("[%s] Converting audio files..." % args.output_json)
dataset = parallel_preprocess(dataset=dataset,
                              input_dir=args.input_dir,
                              dest_dir=args.dest_dir,
                              target_sr=args.target_sr,
                              speed=args.speed,
                              overwrite=args.overwrite,
                              parallel=args.parallel)

print("[%s] Generating json..." % args.output_json)
df = pd.DataFrame(dataset, dtype=object)

# Save json with python. df.to_json() produces back slashed in file paths
dataset = df.to_dict(orient='records')
with open(args.output_json, 'w') as fp:
    json.dump(dataset, fp, indent=2)
