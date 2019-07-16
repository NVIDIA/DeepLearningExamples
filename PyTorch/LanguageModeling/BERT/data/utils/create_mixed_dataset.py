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
import multiprocessing as mp
"""
mixing hdf5 shards with each other
"""
  

def shard_files(output_files, l_instance_ids, lookuptable, files):
 
    l_input_ids = []
    l_input_masks = []
    l_segment_ids = []
    l_masked_lm_positions = []
    l_masked_lm_ids = []
    l_next_sentence_labels = []

    seq_len = 0
    pred_len = 0
    with h5py.File(files[0], 'r') as f:
      seq_len = f['input_ids'].shape[1]
      pred_len = f['masked_lm_positions'].shape[1]

    assert(seq_len > 0 and pred_len > 0)
    for i, output_file in enumerate(output_files):
      output_length = len(l_instance_ids[i])
      print("preparing to write {} instances to {}".format(output_length, output_file))
      input_ids = np.ones([output_length, seq_len], dtype=np.int32)
      input_masks = np.ones([output_length, seq_len], dtype=np.int8)
      segment_ids = np.ones([output_length, seq_len], dtype=np.int8)
      masked_lm_positions = np.ones([output_length, pred_len], dtype=np.int32)
      masked_lm_ids= np.ones([output_length, pred_len], dtype=np.int32)
      next_sentence_labels = np.ones(output_length, dtype=np.int8)
      l_input_ids.append(input_ids)
      l_input_masks.append(input_masks)
      l_segment_ids.append(segment_ids)
      l_masked_lm_positions.append(masked_lm_positions)
      l_masked_lm_ids.append(masked_lm_ids)
      l_next_sentence_labels.append(next_sentence_labels)
    for did, f in enumerate(tqdm(files)):
      h5_f = h5py.File(f, 'r')
      f_input_ids = h5_f['input_ids'][:]
      f_input_masks = h5_f['input_mask'][:]
      f_segment_ids = h5_f['segment_ids'][:]
      f_masked_lm_positions = h5_f['masked_lm_positions'][:]
      f_masked_lm_ids = h5_f['masked_lm_ids'][:]
      f_next_sentence_labels = h5_f['next_sentence_labels'][:]
      h5_f.close()
      for out_i, out_file in enumerate(output_files):
        instance_ids = l_instance_ids[out_i]
        for l, idx in enumerate(instance_ids):
          doc_id, line_id = lookuptable[idx]
          if doc_id == did:
            l_input_ids[out_i][l] = f_input_ids[line_id]
            l_input_masks[out_i][l] = f_input_masks[line_id]
            l_segment_ids[out_i][l] = f_segment_ids[line_id]
            l_masked_lm_positions[out_i][l] = f_masked_lm_positions[line_id]
            l_masked_lm_ids[out_i][l] = f_masked_lm_ids[line_id]
            l_next_sentence_labels[out_i][l] = f_next_sentence_labels[line_id]
    for out_i, out_file in enumerate(output_files):
      output_length = len(l_input_ids[out_i])
      print("writing {} instances to {}".format(output_length, out_file))
      with h5py.File(out_file, 'w') as f:
        f.create_dataset("input_ids", data=l_input_ids[out_i], dtype='i4', compression='gzip')
        f.create_dataset("input_mask", data=l_input_masks[out_i], dtype='i1', compression='gzip')
        f.create_dataset("segment_ids", data=l_segment_ids[out_i], dtype='i1', compression='gzip')
        f.create_dataset("masked_lm_positions", data=l_masked_lm_positions[out_i], dtype='i4', compression='gzip')
        f.create_dataset("masked_lm_ids", data=l_masked_lm_ids[out_i], dtype='i4', compression='gzip')
        f.create_dataset("next_sentence_labels", data=l_next_sentence_labels[out_i], dtype='i1', compression='gzip')

    
def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_files",
                        default=None,
                        type=str,
                        required=True,
                        help="comma seperated list of file paths, each path can be either file or directory of files")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="directory for output shards")
    parser.add_argument("--lookup",
                        default=None,
                        type=str,
                        required=True,
                        help="path to lookup table")
    parser.add_argument("--indices_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="path to shuffled instance indices")
    parser.add_argument("--index_range",
                        default=None,
                        type=str,
                        required=True,
                        help="index range of output files to be written out, e.g specify '0-100' for writing out 0.hdf5 , ..., 100.hdf5")
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



    print("loading indices file")
    start_idx, end_idx= int(args.index_range.split('-')[0]), int(args.index_range.split('-')[1])
    index_files = []
    instance_ids = []
    for i in range(start_idx, end_idx + 1):
      index_files.append(os.path.join(args.indices_dir, "indices_" + str(i) + ".npy"))
      instance_ids.append( np.load(index_files[-1]))

    output_files = [os.path.join(args.output_dir, indices_file.split('.')[0].split('_')[-1] + ".hdf5") for indices_file in index_files]
    print("output_files", output_files)

    print("loading lookup table")
    lookup_table = np.load(args.lookup)
    shard_files(output_files, instance_ids, lookup_table, input_files)



if __name__ == "__main__":
    main()

