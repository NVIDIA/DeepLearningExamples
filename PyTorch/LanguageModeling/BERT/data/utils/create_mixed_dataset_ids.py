from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open
import h5py
import numpy as np
from tqdm import tqdm, trange
import random
import collections
import math
from tqdm import tqdm
import multiprocessing as mp
import pickle
import json
"""
mixing hdf5 shards with each other
"""
def load_and_prepare(input_files, num_shards):

  seq_len = None
  pred_len = None

  input_lengths = []
  for input_file in input_files:
    with h5py.File(input_file, 'r') as f:
      input_lengths.append(len(f['input_ids']))
      if seq_len is None:
        seq_len = f['input_ids'].shape[1]
        pred_len = f['masked_lm_ids'].shape[1]

  assert (isinstance(seq_len, int) and isinstance(pred_len, int))

  total_instances = sum(input_lengths)
  n_inst_per_file = math.ceil(total_instances * 1.0 / num_shards)
  permutation = np.random.permutation(total_instances)


  instance_indices = []
  for i in range(0, num_shards):
    start_pos = i * n_inst_per_file
    end_pos = min((i+1) * n_inst_per_file, total_instances)
    instance_indices.append(permutation[start_pos:end_pos])

  return seq_len, pred_len, input_lengths, instance_indices



    
def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_files",
                        default=None,
                        type=str,
                        required=True,
                        help="comma seperated list of file paths, each path can be either file or directory of hdf5 files")
    parser.add_argument("--num_output_shards",
                        default=None,
                        type=int,
                        required=True,
                        help="number of shards to be created. shards will be created as even as possible.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="directory for meta files")
    parser.add_argument('--random_seed',
                        type=int,
                        default=12345,
                        help="random seed for initialization")

    args = parser.parse_args()

    rng = random.Random(args.random_seed)
    np.random.seed(args.random_seed)


    input_paths = args.input_files.strip().split(',')
    input_paths = [f for f in input_paths if f]

    input_files = []
    for path in input_paths:
      if os.path.isfile(path):
        assert (path.endswith('.hdf5')), "file must be hdf5 file"
        input_files.append(path)
      else:
        assert os.path.isdir(path)
        hdf5_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.hdf5')]
        input_files.extend(hdf5_files)
    input_files.sort()

    assert(os.path.isdir(args.output_dir))

    print("load and prepare")
    seq_len, pred_len, input_lengths, output_inst_indices = load_and_prepare(input_files, args.num_output_shards)
    print("preparing lookup table")
    total_num_instances = sum(input_lengths)
    out_2_in = dict()
    length_so_far = 0
    for i, l in enumerate(input_lengths):
      for j in range(l):
        out_2_in[length_so_far + j] = (i, j)
      length_so_far += input_lengths[i]


    
    output_files = [os.path.join(args.output_dir, "indices_" + str(i) + ".npy") for i in range(args.num_output_shards)]
    print("save data")


    with open(os.path.join(args.output_dir, 'lookup_table.pkl'), 'wb') as f:
      pickle.dump(out_2_in, f)

    for i, out_file in enumerate(output_files):
      np.save(out_file, output_inst_indices[i])
    

    meta = {'seq_len': seq_len, 'pred_len':pred_len}

    with open(os.path.join(args.output_dir, 'meta_data.pkl'), 'wb') as f:
      pickle.dump(meta, f)

    

    



if __name__ == "__main__":
    main()
