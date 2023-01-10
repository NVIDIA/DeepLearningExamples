# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


# Copyright 2020 Chengxi Zang
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import os
import pandas as pd
import argparse
import time

from moflow.config import CONFIGS
from moflow.data.data_frame_parser import DataFrameParser
from moflow.data.encoding import MolEncoder


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str,
                        choices=list(CONFIGS),
                        help='dataset to be downloaded')
    parser.add_argument('--data_dir', type=str, default='/data')
    args = parser.parse_args()
    return args

def main(args):
    start_time = time.time()
    args = parse_args()
    print('args', vars(args))

    assert args.data_name in CONFIGS
    dataset_config = CONFIGS[args.data_name].dataset_config

    preprocessor = MolEncoder(out_size=dataset_config.max_num_atoms)
    
    input_path = os.path.join(args.data_dir, dataset_config.csv_file)
    output_path = os.path.join(args.data_dir, dataset_config.dataset_file)

    print(f'Preprocessing {args.data_name} data:')
    df = pd.read_csv(input_path, index_col=0)
    parser = DataFrameParser(preprocessor, labels=dataset_config.labels, smiles_col=dataset_config.smiles_col)
    dataset = parser.parse(df)

    dataset.save(output_path)
    print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
