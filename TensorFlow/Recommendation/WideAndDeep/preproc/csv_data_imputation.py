# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function

import pandas as pd
import os
import glob
import tqdm
import argparse
from joblib import Parallel, delayed


parser = argparse.ArgumentParser()
parser.add_argument('--train_files_pattern', default='train_feature_vectors_integral_eval.csv/part-*')
parser.add_argument('--valid_files_pattern', default='validation_feature_vectors_integral.csv/part-*')
parser.add_argument('--train_dst_dir', default='train_feature_vectors_integral_eval_imputed.csv')
parser.add_argument('--valid_dst_dir', default='validation_feature_vectors_integral_imputed.csv')
parser.add_argument('--header_path', default='train_feature_vectors_integral_eval.csv.header')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

header = pd.read_csv(args.header_path, header=None)
columns = header[0].to_list()

train_files = glob.glob(args.train_files_pattern)
print('train files: ', train_files)

def get_counts(f):
    df = pd.read_csv(f, header=None, dtype=object, names=columns, na_values='None')
    counts = {}
    for c in df:
        counts[c] = df[c].value_counts()
    return counts

all_counts = Parallel(n_jobs=args.num_workers)(delayed(get_counts)(f) for f in train_files)
cols = len(all_counts[0])
imputation_dict = {}
for c in tqdm.tqdm(columns):
    temp = None
    for i in range(len(all_counts)):
        if temp is None:
            temp = pd.Series(all_counts[i][c])
        else:
            temp += pd.Series(all_counts[i][c])
    if len(temp) == 0:
        imputation_dict[c] = 0
    else:
        imputation_dict[c] = temp.index[0]

print('imputation_dict: ', imputation_dict)

if not os.path.exists(args.train_dst_dir):
    os.mkdir(args.train_dst_dir)

def impute_part(src_path, dst_dir):
    print('imputing: ', src_path, ' to: ', dst_dir)
    filename = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, filename)

    df = pd.read_csv(src_path, header=None, dtype=object, names=columns, na_values='None')
    df2 = df.fillna(imputation_dict)
    df2.to_csv(dst_path, header=None, index=None)


print('launching imputation for train CSVs')
Parallel(n_jobs=args.num_workers)(delayed(impute_part)(f, args.train_dst_dir) for f in train_files)

valid_files = glob.glob(args.valid_files_pattern)

if not os.path.exists(args.valid_dst_dir):
    os.mkdir(args.valid_dst_dir)

print('launching imputation for validation CSVs')
Parallel(n_jobs=args.num_workers)(delayed(impute_part)(f, args.valid_dst_dir) for f in valid_files)

print('Done!')
