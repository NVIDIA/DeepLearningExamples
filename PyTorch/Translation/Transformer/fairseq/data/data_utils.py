# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import contextlib
import itertools
import os

import numpy as np
import torch

import fairseq.data.batch_C
import sys
from .dictionary import Dictionary


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


def load_dictionaries(args):
    if args.source_lang is None or args.target_lang is None:
        args.source_lang, args.target_lang = infer_language_pair(args.data)
    if args.source_lang is None or args.target_lang is None:
        raise Exception('Could not infer language pair, please provide it explicitly')

    # load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.target_lang)))
    assert src_dict.pad() == tgt_dict.pad()
    assert src_dict.eos() == tgt_dict.eos()
    assert src_dict.unk() == tgt_dict.unk()
    args.src_vocab_size = len(src_dict)
    args.tgt_vocab_size = len(tgt_dict)
    args.padding_idx = src_dict.pad()
    print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
    print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
    return src_dict, tgt_dict


class ShardedIterator(object):
    """A sharded wrapper around an iterable (padded to length)."""

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value,
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count."""

    def __init__(self, iterable):
        self.iterable = iterable
        self.count = 0
        self.itr = iter(self)

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        for x in self.iterable:
            self.count += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        return self.count < len(self)

    def skip(self, num_to_skip):
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False, pad_sequence=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    #size = max(v.size(0) for v in values)
    orig_size = max(v.size(0) for v in values)
    size = 0
    if pad_sequence > 1:
        size = orig_size // pad_sequence * pad_sequence
        if orig_size % pad_sequence > 0:
            size += pad_sequence
    else:
        size = orig_size
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False, pad_sequence=1):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            pad_sequence,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'prev_output_tokens': prev_output_tokens,
        },
        'target': target,
    }


def get_dummy_batch(num_tokens, src_dict, tgt_dict, src_len=128, tgt_len=128,
                    left_pad_source=True, left_pad_target=False, pad_sequence=1):
    bsz = num_tokens // max(src_len, tgt_len)
    dummy_samples = [
        {
            'id': i,
            'source': src_dict.dummy_sentence(src_len),
            'target': tgt_dict.dummy_sentence(tgt_len) if tgt_dict is not None else None,
        }
        for i in range(bsz)
    ]
    return collate(
        dummy_samples, pad_idx=src_dict.pad(), eos_idx=src_dict.eos(),
        left_pad_source=left_pad_source, left_pad_target=left_pad_target,
        pad_sequence=pad_sequence,
    )


class EpochBatchIterator(object):
    """Iterate over a FairseqDataset and yield batches bucketed by size.

    Batches may contain sequences of different lengths. This iterator can be
    reused across multiple epochs with the next_epoch_itr() method.

    Args:
        dataset: a FairseqDataset
        max_tokens: max number of tokens in each batch
        max_sentences: max number of sentences in each batch
        max_positions: max sentence length supported by the model
        required_batch_size_multiple: require batch size to be a multiple of N
        seed: seed for random number generator for reproducibility
        num_shards: shard the data iterator into N shards
        shard_id: which shard of the data iterator to return
    """

    def __init__(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        required_batch_size_multiple=1, seed=1,
        num_shards=1, shard_id=0, epoch=0
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.max_sentences = max_sentences if max_sentences is not None else float('Inf')
        self.max_positions = max_positions
        self.bsz_mult = required_batch_size_multiple
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.epoch = epoch
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

        with numpy_seed(self.seed):
            indices = self.dataset.ordered_indices(self.seed, self.epoch)
#need integer, rather than float('Inf') values
            max_sentences = max_sentences if max_sentences is not None else sys.maxsize
            max_positions_num = 1024
            max_tokens = max_tokens if max_tokens is not None else sys.maxsize
            #Following line is workaround due to the fact we cannot pass None object as argument
            tgt_sizes = self.dataset.tgt_sizes if self.dataset.tgt_sizes is not None else self.dataset.src_sizes
            batches = fairseq.data.batch_C.make_batches(
                self.dataset.src_sizes, tgt_sizes, indices, max_tokens,
                max_sentences, self.bsz_mult, max_positions_num)
            self.frozen_batches = tuple(batches)

    def __len__(self):
        return len(self.frozen_batches)

    def next_epoch_itr(self, shuffle=True):
        """Shuffle batches and return a new iterator over the dataset."""
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(self.epoch, shuffle)
        return self._cur_epoch_itr

    def end_of_epoch(self):
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            itr = self._get_iterator_for_epoch(self.epoch, state_dict.get('shuffle', True))
            if itr_pos < len(itr):
                self._next_epoch_itr = itr.skip(itr_pos)

    def _get_iterator_for_epoch(self, epoch, shuffle):
        if shuffle:
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with numpy_seed(self.seed + epoch):
                batches = list(self.frozen_batches)  # copy
                np.random.shuffle(batches)
        else:
            batches = self.frozen_batches
        return CountingIterator(torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collater,
            num_workers=1,
            batch_sampler=ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[]),
        ))


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
