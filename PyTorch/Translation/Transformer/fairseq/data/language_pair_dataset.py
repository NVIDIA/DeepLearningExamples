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

import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from . import data_utils
import itertools
import os
import sys
from fairseq.data import IndexedInMemoryDataset, IndexedRawTextDataset


class LanguagePairDataset(Dataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        pad_sequence=1, shuffle=True,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.pad_sequence = pad_sequence
        self.shuffle = shuffle
        print("| Sentences are being padded to multiples of: {}".format(self.pad_sequence), file=sys.stderr)

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return data_utils.collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            pad_sequence=self.pad_sequence,
        )

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        orig_size = max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        assert self.pad_sequence > 0, "Padding multiple has to be greater than 0"
        size = 0
        if self.pad_sequence > 1:
            size = orig_size // self.pad_sequence * self.pad_sequence
            if orig_size % self.pad_sequence > 0:
                size += self.pad_sequence
        else:
            size = orig_size
        return size
        #return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self, seed=None, epoch=1):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.RandomState(seed + epoch).permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions and
            (self.tgt_sizes is None or self.tgt_sizes[index] <= max_target_positions)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions
        assert len(max_positions) == 2
        max_src_pos, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_target_positions, max_tgt_pos)


def load_dataset(args, datasets, split, src_dict, tgt_dict, combine=False):
    """Load a dataset split."""

    def split_exists(split, src, tgt, lang):
        filename = os.path.join(args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        if args.raw_text and IndexedRawTextDataset.exists(filename):
            return True
        elif not args.raw_text and IndexedInMemoryDataset.exists(filename):
            return True
        return False

    def indexed_dataset(path, dictionary):
        if args.raw_text:
            return IndexedRawTextDataset(path, dictionary)
        elif IndexedInMemoryDataset.exists(path):
            return IndexedInMemoryDataset(path, fix_lua_indexing=True)
        return None

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        src, tgt = args.source_lang, args.target_lang
        if split_exists(split_k, src, tgt, src):
            prefix = os.path.join(args.data, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src):
            prefix = os.path.join(args.data, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, args.data))

        src_datasets.append(indexed_dataset(prefix + src, src_dict))
        tgt_datasets.append(indexed_dataset(prefix + tgt, tgt_dict))

        print('| {} {} {} examples'.format(args.data, split_k, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        src_sizes = src_dataset.sizes
        tgt_sizes = tgt_dataset.sizes
    else:
        src_dataset = ConcatDataset(src_datasets)
        tgt_dataset = ConcatDataset(tgt_datasets)
        src_sizes = np.concatenate([ds.sizes for ds in src_datasets])
        tgt_sizes = np.concatenate([ds.sizes for ds in tgt_datasets])

    datasets[split] = LanguagePairDataset(
        src_dataset, src_sizes, src_dict,
        tgt_dataset, tgt_sizes, tgt_dict,
        left_pad_source=args.left_pad_source,
        left_pad_target=args.left_pad_target,
        max_source_positions=args.max_source_positions,
        max_target_positions=args.max_target_positions,
        pad_sequence=args.pad_sequence,
    )


def load_dataset_splits(args, splits, src_dict, tgt_dict):
    datasets = {}
    for split in splits:
        if split == 'train':
            load_dataset(args, datasets, split, src_dict, tgt_dict, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    load_dataset(args, datasets, split_k, src_dict, tgt_dict, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e
    return datasets
