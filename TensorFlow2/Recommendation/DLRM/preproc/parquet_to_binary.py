# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
import glob
import argparse
import tqdm
import subprocess

def process_file(f, dst):

    all_columns_sorted = [f'_c{i}' for i in range(0, 40)]

    data = pd.read_parquet(f)
    data = data[all_columns_sorted]

    dense_columns = [f'_c{i}' for i in range(1, 14)]
    data[dense_columns] = data[dense_columns].astype(np.float32)

    data = data.to_records(index=False)
    data = data.tobytes()

    dst_file = dst + '/' + f.split('/')[-1] + '.bin'
    with open(dst_file, 'wb') as dst_fd:
        dst_fd.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--intermediate_dir', type=str)
    parser.add_argument('--dst_dir', type=str)
    parser.add_argument('--parallel_jobs', default=40, type=int)
    args = parser.parse_args()

    print('Processing train files...')
    train_src_files = glob.glob(args.src_dir + '/train/*.parquet')
    train_intermediate_dir = args.intermediate_dir + '/train'
    os.makedirs(train_intermediate_dir, exist_ok=True)

    Parallel(n_jobs=args.parallel_jobs)(delayed(process_file)(f, train_intermediate_dir) for f in tqdm.tqdm(train_src_files))

    print('Train files conversion done')

    print('Processing test files...')
    test_src_files = glob.glob(args.src_dir + '/test/*.parquet')
    test_intermediate_dir = args.intermediate_dir + '/test'
    os.makedirs(test_intermediate_dir, exist_ok=True)

    Parallel(n_jobs=args.parallel_jobs)(delayed(process_file)(f, test_intermediate_dir) for f in tqdm.tqdm(test_src_files))
    print('Test files conversion done')

    print('Processing validation files...')
    valid_src_files = glob.glob(args.src_dir + '/validation/*.parquet')
    valid_intermediate_dir = args.intermediate_dir + '/valid'
    os.makedirs(valid_intermediate_dir, exist_ok=True)

    Parallel(n_jobs=args.parallel_jobs)(delayed(process_file)(f, valid_intermediate_dir) for f in tqdm.tqdm(valid_src_files))
    print('Validation files conversion done')

    os.makedirs(args.dst_dir, exist_ok=True)

    print('Concatenating train files')
    os.system(f'cat {train_intermediate_dir}/*.bin > {args.dst_dir}/train_data.bin')

    print('Concatenating test files')
    os.system(f'cat {test_intermediate_dir}/*.bin > {args.dst_dir}/test_data.bin')

    print('Concatenating validation files')
    os.system(f'cat {valid_intermediate_dir}/*.bin > {args.dst_dir}/val_data.bin')
    print('Done')


if __name__ == '__main__':
    main()
