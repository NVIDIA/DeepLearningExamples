import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from lddl.utils import expand_outdir_and_mkdir, get_all_files_paths_under


def collect_data(args):
  npzs = [
      fp for fp in get_all_files_paths_under(args.in_dir)
      if 'lens_' in fp and os.path.splitext(fp)[1] == '.npz'
  ]
  min_lens, max_lens = {}, {}
  seq_len_hist, padded_zero_hist = None, None
  for npz in npzs:
    rank = int(os.path.splitext(os.path.basename(npz))[0].split('_')[1])
    with np.load(npz) as data:
      min_lens[rank] = data['min_lens']
      max_lens[rank] = data['max_lens']
      if seq_len_hist is None:
        seq_len_hist = data['seq_len_hist']
      else:
        seq_len_hist += data['seq_len_hist']
      if padded_zero_hist is None:
        padded_zero_hist = data['padded_zero_hist']
      else:
        padded_zero_hist += data['padded_zero_hist']
    assert max_lens[rank].shape == min_lens[rank].shape

  return min_lens, max_lens, seq_len_hist, padded_zero_hist


def plot_rank_diff(args, min_lens, max_lens):
  """ Make sure the diff between min seq lens and max seq lens is smaller than
  the bin size.

  min_lens and max_lens and dict[int] -> np.array that map rank number to
  the list of max and min seq lens of all training iterations.
  """
  rank_arrays = []
  diffs = []
  ranks = list(sorted(min_lens.keys()))
  for rank in ranks:
    diffs.append(max_lens[rank] - min_lens[rank])
    rank_arrays.append(np.full(min_lens[rank].shape, rank, dtype=np.uint16))
  rank_arrays = np.concatenate(rank_arrays)
  diffs = np.concatenate(diffs)
  plt.scatter(rank_arrays, diffs, s=0.1)
  plt.xlabel('rank')
  plt.xticks(ranks)
  plt.ylabel('diff')
  plt.yticks(np.arange(0, diffs.max() + 1, 1))
  plt.title('rank vs. diff')
  plt.grid()
  plt.savefig(os.path.join(args.out_dir, 'rank_dist.png'))
  plt.close()


def plot_min_max_lens(args, min_lens, max_lens):
  """ Make sure the min and max seq lens are limited by the bin size.
  """
  ranks = list(sorted(min_lens.keys()))
  for rank in ranks:
    plt.scatter(min_lens[rank], max_lens[rank], s=0.1)
    plt.xlabel('min_lens')
    plt.xticks(np.arange(0, min_lens[rank].max() + args.bin_size,
                         args.bin_size))
    plt.ylabel('max_lens')
    plt.yticks(np.arange(0, max_lens[rank].max() + args.bin_size,
                         args.bin_size))
    plt.title('min_lens vs. max_lens')
    plt.grid()
    plt.savefig(os.path.join(args.out_dir, 'min_max_lens_{}.png'.format(rank)))
    plt.close()


def plot_global_diff(args, min_lens, max_lens):
  """ Make sure that each rank chooses the same bin in each iteration.
  """
  ranks = list(sorted(min_lens.keys()))
  global_min_lens = np.stack([min_lens[rank] for rank in ranks], axis=-1)
  global_max_lens = np.stack([max_lens[rank] for rank in ranks], axis=-1)
  diffs = global_max_lens.max(axis=-1) - global_min_lens.min(axis=-1)
  plt.scatter(np.full(diffs.shape, 0, dtype=np.uint8), diffs, s=0.1)
  plt.xticks([0])
  plt.ylabel('diff')
  plt.yticks(np.arange(0, diffs.max() + 1, 1))
  plt.title('global diff')
  plt.grid()
  plt.savefig(os.path.join(args.out_dir, 'global_diff.png'))
  plt.close()


def plot_seq_len_hist(args, seq_len_hist):
  hist = []
  xticks = []
  for start in range(1, seq_len_hist.shape[0], args.seq_len_hist_bin):
    n = 0
    for seq_len in range(start, start + args.seq_len_hist_bin):
      n += seq_len_hist[seq_len]
    hist.append(n)
    xticks.append('{}-{}'.format(start, start + args.seq_len_hist_bin - 1))
  plt.figure(figsize=(20, 5))
  plt.bar(xticks, hist)
  plt.xlabel('seq_lens')
  plt.ylabel('# Samples')
  plt.title('Sequence Length Histogram')
  plt.grid()
  plt.savefig(os.path.join(args.out_dir, 'seq_len_hist.png'))
  plt.close()


def plot_padded_zero_hist(args, padded_zero_hist):
  plt.bar(np.arange(0, len(padded_zero_hist)), padded_zero_hist)
  plt.xlabel('# zeros in a sequence')
  plt.ylabel('# samples')
  plt.title('# zeros in a sequence vs. # samples')
  plt.grid()
  plt.savefig(os.path.join(args.out_dir, 'padded_zero_hist.png'))
  plt.close()


def hist_sum(hist):
  s = 0
  for v in range(hist.shape[0]):
    s += v * hist[v]
  return s


def calculate_padded_zero_ratio(padded_zero_hist, seq_len_hist):
  num_zeros = hist_sum(padded_zero_hist)
  num_tokens = hist_sum(seq_len_hist)
  print('padded_zeros : tokens = {} : {} = {} : 1'.format(
      num_zeros, num_tokens, num_zeros / num_tokens))


def main(args):
  args.out_dir = expand_outdir_and_mkdir(args.out_dir)
  min_lens, max_lens, seq_len_hist, padded_zero_hist = collect_data(args)
  plot_rank_diff(args, min_lens, max_lens)
  plot_min_max_lens(args, min_lens, max_lens)
  plot_global_diff(args, min_lens, max_lens)
  plot_seq_len_hist(args, seq_len_hist)
  plot_padded_zero_hist(args, padded_zero_hist)
  calculate_padded_zero_ratio(padded_zero_hist, seq_len_hist)


def attach_args(parser=argparse.ArgumentParser()):
  parser.add_argument('--in-dir', type=str, required=True)
  parser.add_argument('--out-dir', type=str, default="./fig")
  parser.add_argument('--bin-size', type=int, default=32)
  parser.add_argument('--seq-len-hist-bin', type=int, default=32)

  return parser


if __name__ == "__main__":
  main(attach_args().parse_args())
