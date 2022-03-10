import argparse
import logging
import numpy as np
import os
import random
import time
import torch
from transformers import BertTokenizerFast

from lddl.torch import get_bert_pretrain_data_loader
from lddl.torch.utils import barrier, get_rank
from lddl.utils import mkdir


def get_batch_seq_lens(attention_mask):
  return attention_mask.sum(dim=1).int()


class AverageMeter:
  """
  Computes and stores the average and current value
  """

  def __init__(self, warmup=0, keep=False):
    self.reset()
    self.warmup = warmup
    self.keep = keep

  def reset(self):
    self.val = 0
    self.avg = 0
    self.max = float('-inf')
    self.min = float('inf')
    self.sum = 0
    self.count = 0
    self.iters = 0
    self.vals = []

  def update(self, val, n=1):
    self.iters += 1
    self.val = val

    if self.iters > self.warmup:
      self.sum += val * n
      self.max = max(val, self.max)
      self.min = min(val, self.min)
      self.count += n
      self.avg = self.sum / self.count
      if self.keep:
        self.vals.append(val)


class Histogram:
  """
  Computes and stores the histogram of values.
  """

  def __init__(self):
    self.hist = np.zeros((1,), dtype=np.uint64)

  def update(self, val, n=1):
    if val >= self.hist.shape[0]:
      new_hist = np.zeros((val + 1,), dtype=np.uint64)
      new_hist[:self.hist.shape[0]] = self.hist[:]
      self.hist = new_hist
    self.hist[val] += n

  def update_with_tensor(self, t):
    for v in t.flatten().tolist():
      self.update(v)


def main(args):
  torch.cuda.set_device(args.local_rank)
  world_size = int(os.getenv('WORLD_SIZE', 1))
  if world_size > 1:
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

  if get_rank() == 0 and args.seq_len_dir is not None:
    mkdir(args.seq_len_dir)

  loader = get_bert_pretrain_data_loader(
      args.path,
      local_rank=args.local_rank,
      shuffle_buffer_size=args.shuffle_buffer_size,
      shuffle_buffer_warmup_factor=args.shuffle_buffer_warmup_factor,
      vocab_file=args.vocab_file,
      data_loader_kwargs={
          'batch_size': args.batch_size,
          'num_workers': args.workers,
          'prefetch_factor': args.prefetch
      },
      mlm_probability=args.mlm_probability,
      base_seed=args.seed,
      log_dir=args.log_dir,
      log_level=getattr(logging, args.log_level),
      return_raw_samples=args.debug,
      start_epoch=args.start_epoch,
      sequence_length_alignment=args.sequence_length_alignment,
      ignore_index=args.ignore_index,
  )
  if os.path.isfile(args.vocab_file):
    test_tokenizer = BertTokenizerFast(args.vocab_file)
  else:
    test_tokenizer = BertTokenizerFast.from_pretrained(args.vocab_file)

  meter = AverageMeter(warmup=args.warmup)

  lens_shape = (args.epochs, min(len(loader), args.iters_per_epoch))
  min_lens, max_lens, batch_sizes, padded_lens = (
      np.zeros(lens_shape, dtype=np.uint16),
      np.zeros(lens_shape, dtype=np.uint16),
      np.zeros(lens_shape, dtype=np.uint16),
      np.zeros(lens_shape, dtype=np.uint16),
  )
  seq_len_hist = Histogram()
  padded_zero_hist = Histogram()

  for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    barrier()
    epoch_timer_start = time.time()
    batch_timer_start = time.time()
    total_samples = 0
    for i, data in enumerate(loader):
      if i >= args.iters_per_epoch:
        break
      if not args.debug:
        (input_ids, token_type_ids, attention_mask, labels,
         next_sentence_labels) = (
             data['input_ids'],
             data['token_type_ids'],
             data['attention_mask'],
             data['labels'],
             data['next_sentence_labels'],
         )
      batch_timer_stop = time.time()
      elapsed = batch_timer_stop - batch_timer_start
      meter.update(elapsed)

      if args.debug:
        current_samples = len(data[0]) * world_size
      else:
        current_samples = input_ids.size(0) * world_size
        assert input_ids.size() == token_type_ids.size()
        assert input_ids.size() == attention_mask.size()
        assert input_ids.size() == labels.size()
        assert next_sentence_labels.dim() == 1
        assert input_ids.size(0) == next_sentence_labels.size(0)
        seq_lens = get_batch_seq_lens(attention_mask)
        seq_len_hist.update_with_tensor(seq_lens)
        (
            min_lens[epoch - args.start_epoch, i],
            max_lens[epoch - args.start_epoch, i],
        ) = seq_lens.min(), seq_lens.max()
        batch_sizes[epoch - args.start_epoch, i] = input_ids.size(0)
        padded_lens[epoch - args.start_epoch, i] = input_ids.size(1)
        padded_zero_hist.update_with_tensor(input_ids.size(1) - seq_lens)

      total_samples += current_samples
      current_throughput = current_samples / elapsed
      if (i + 1) % args.log_freq == 0 and get_rank() == 0:
        avg_throughput = total_samples / meter.sum
        print('avg_throughput={}, avg_latency={} ms, '
              'min_latency={} ms, max_latency={} ms, '
              'current_throughput={}, current_latency={} ms'.format(
                  avg_throughput,
                  meter.avg * 1000,
                  meter.min * 1000,
                  meter.max * 1000,
                  current_throughput,
                  elapsed * 1000,
              ))
        if args.debug:
          print('len(data[0])={}'.format(len(data[0])))
          print('sample=({} <SEP> {} - {})'.format(
              data[0][0],
              data[1][0],
              data[2][0],
          ))
        else:
          print("Min length={} Max length={} Diff={}".format(
              min_lens[epoch - args.start_epoch, i],
              max_lens[epoch - args.start_epoch, i],
              max_lens[epoch - args.start_epoch, i] -
              min_lens[epoch - args.start_epoch, i],
          ))
          print('input_ids.size()={}'.format(input_ids.size()))
          print('input_ids[0]={}'.format(input_ids[0]))
          print('convert_ids_to_tokens(input_ids[0])={}'.format(
              test_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())))
          print('token_type_ids[0]={}'.format(token_type_ids[0]))
          print('attention_mask[0]={}'.format(attention_mask[0]))
          print('labels[0]={}'.format(labels[0]))
          print('next_sentence_labels[0]={}'.format(next_sentence_labels[0]))
          mask = labels[0] != args.ignore_index
          input_ids[0, mask] = labels[0, mask]
          print('original sequence={}'.format(
              test_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())))
      barrier()
      batch_timer_start = time.time()
    epoch_timer_stop = time.time()
    epoch_elapsed = epoch_timer_stop - epoch_timer_start
    if args.local_rank == 0:
      avg_throughput = total_samples / meter.sum
      print('epoch={}, epoch_elapsed={}, avg_throughput={}, '
            'total_samples={}'.format(
                epoch,
                epoch_elapsed,
                avg_throughput,
                total_samples,
            ))
    assert meter.iters == min(len(loader), args.iters_per_epoch)
    meter.reset()

  if args.seq_len_dir is not None:
    # Save the sequence lengths to file
    np.savez_compressed(
        os.path.join(args.seq_len_dir, 'lens_{}.npz'.format(get_rank())),
        min_lens=min_lens,
        max_lens=max_lens,
        batch_sizes=batch_sizes,
        padded_lens=padded_lens,
        seq_len_hist=seq_len_hist.hist,
        padded_zero_hist=padded_zero_hist.hist,
    )


def attach_args(parser=argparse.ArgumentParser()):
  parser.add_argument('--path', type=str, required=True)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--workers', type=int, default=4)
  parser.add_argument('--warmup', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--iters-per-epoch', type=int, default=float('inf'))
  parser.add_argument('--prefetch', type=int, default=2)
  parser.add_argument(
      '--local_rank',
      type=int,
      default=os.getenv('LOCAL_RANK', 0),
  )
  parser.add_argument('--mlm-probability', type=float, default=0.15)
  parser.add_argument('--shuffle-buffer-size', type=int, default=16384)
  parser.add_argument('--shuffle-buffer-warmup-factor', type=int, default=16)
  parser.add_argument('--vocab-file', type=str, required=True)
  parser.add_argument('--seed', type=int, default=127)
  parser.add_argument('--start-epoch', type=int, default=0)
  parser.add_argument('--debug', action='store_true', default=False)
  parser.add_argument('--log-freq', type=int, default=1000)
  parser.add_argument('--log-dir', type=str, default=None)
  parser.add_argument(
      '--log-level',
      type=str,
      choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
      default='WARNING',
  )
  parser.add_argument('--seq-len-dir', type=str, default=None)
  parser.add_argument('--sequence-length-alignment', type=int, default=8)
  parser.add_argument('--ignore-index', type=int, default=-1)
  return parser


if __name__ == '__main__':
  main(attach_args().parse_args())
