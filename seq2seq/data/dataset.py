import logging
from operator import itemgetter

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import seq2seq.data.config as config
from seq2seq.data.sampler import BucketingSampler
from seq2seq.data.sampler import DistributedSampler
from seq2seq.data.sampler import ShardingSampler
from seq2seq.data.sampler import StaticDistributedSampler


def build_collate_fn(batch_first=False, parallel=True, sort=False):
    """
    Factory for collate_fn functions.

    :param batch_first: if True returns batches in (batch, seq) format, if
        False returns in (seq, batch) format
    :param parallel: if True builds batches from parallel corpus (src, tgt)
    :param sort: if True sorts by src sequence length within each batch
    """
    def collate_seq(seq):
        """
        Builds batches for training or inference.
        Batches are returned as pytorch tensors, with padding.

        :param seq: list of sequences
        """
        lengths = [len(s) for s in seq]
        batch_length = max(lengths)

        shape = (batch_length, len(seq))
        seq_tensor = torch.full(shape, config.PAD, dtype=torch.int64)

        for i, s in enumerate(seq):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])

        if batch_first:
            seq_tensor = seq_tensor.t()

        return (seq_tensor, lengths)

    def parallel_collate(seqs):
        """
        Builds batches from parallel dataset (src, tgt), optionally sorts batch
        by src sequence length.

        :param seqs: tuple of (src, tgt) sequences
        """
        src_seqs, tgt_seqs = zip(*seqs)
        if sort:
            indices, src_seqs = zip(*sorted(enumerate(src_seqs),
                                            key=lambda item: len(item[1]),
                                            reverse=True))
            tgt_seqs = [tgt_seqs[idx] for idx in indices]

        return tuple([collate_seq(s) for s in [src_seqs, tgt_seqs]])

    def single_collate(src_seqs):
        """
        Builds batches from text dataset, optionally sorts batch by src
        sequence length.

        :param src_seqs: source sequences
        """
        if sort:
            indices, src_seqs = zip(*sorted(enumerate(src_seqs),
                                            key=lambda item: len(item[1]),
                                            reverse=True))
        else:
            indices = range(len(src_seqs))

        return collate_seq(src_seqs), tuple(indices)

    if parallel:
        return parallel_collate
    else:
        return single_collate


class TextDataset(Dataset):
    def __init__(self, src_fname, tokenizer, min_len=None, max_len=None,
                 sort=False, max_size=None):
        """
        Constructor for the TextDataset. Builds monolingual dataset.

        :param src_fname: path to the file with data
        :param tokenizer: tokenizer
        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        :param sort: sorts dataset by sequence length
        :param max_size: loads at most 'max_size' samples from the input file,
            if None loads the entire dataset
        """

        self.min_len = min_len
        self.max_len = max_len
        self.parallel = False
        self.sorted = False

        self.src = self.process_data(src_fname, tokenizer, max_size)

        if min_len is not None and max_len is not None:
            self.filter_data(min_len, max_len)

        lengths = [len(s) for s in self.src]
        self.lengths = torch.tensor(lengths)

        if sort:
            self.sort_by_length()

    def sort_by_length(self):
        """
        Sorts dataset by the sequence length.
        """
        self.lengths, indices = self.lengths.sort(descending=True)

        self.src = [self.src[idx] for idx in indices]
        self.indices = indices.tolist()
        self.sorted = True

    def unsort(self, array):
        """
        "Unsorts" given array (restores original order of elements before
        dataset was sorted by sequence length).

        :param array: array to be "unsorted"
        """
        if self.sorted:
            inverse = sorted(enumerate(self.indices), key=itemgetter(1))
            array = [array[i[0]] for i in inverse]
        return array

    def filter_data(self, min_len, max_len):
        """
        Preserves only samples which satisfy the following inequality:
            min_len <= sample sequence length <= max_len

        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        """
        logging.info(f'Filtering data, min len: {min_len}, max len: {max_len}')

        initial_len = len(self.src)
        filtered_src = []
        for src in self.src:
            if min_len <= len(src) <= max_len:
                filtered_src.append(src)

        self.src = filtered_src
        filtered_len = len(self.src)
        logging.info(f'Pairs before: {initial_len}, after: {filtered_len}')

    def process_data(self, fname, tokenizer, max_size):
        """
        Loads data from the input file.

        :param fname: input file name
        :param tokenizer: tokenizer
        :param max_size: loads at most 'max_size' samples from the input file,
            if None loads the entire dataset
        """
        logging.info(f'Processing data from {fname}')
        data = []
        with open(fname) as dfile:
            for idx, line in enumerate(dfile):
                if max_size and idx == max_size:
                    break
                entry = tokenizer.segment(line)
                entry = torch.tensor(entry)
                data.append(entry)
        return data

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx]

    def get_loader(self, batch_size=1, seeds=None, shuffle=False,
                   num_workers=0, batch_first=False, pad=False,
                   batching=None, batching_opt={}):

        collate_fn = build_collate_fn(batch_first, parallel=self.parallel,
                                      sort=True)

        if shuffle:
            if batching == 'random':
                sampler = DistributedSampler(self, batch_size, seeds)
            elif batching == 'sharding':
                sampler = ShardingSampler(self, batch_size, seeds,
                                          batching_opt['shard_size'])
            elif batching == 'bucketing':
                sampler = BucketingSampler(self, batch_size, seeds,
                                           batching_opt['num_buckets'])
            else:
                raise NotImplementedError
        else:
            sampler = StaticDistributedSampler(self, batch_size, pad)

        return DataLoader(self,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          sampler=sampler,
                          num_workers=num_workers,
                          pin_memory=True,
                          drop_last=False)


class ParallelDataset(TextDataset):
    def __init__(self, src_fname, tgt_fname, tokenizer,
                 min_len, max_len, sort=False, max_size=None):
        """
        Constructor for the ParallelDataset.
        Tokenization is done when the data is loaded from the disk.

        :param src_fname: path to the file with src language data
        :param tgt_fname: path to the file with tgt language data
        :param tokenizer: tokenizer
        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        :param sort: sorts dataset by sequence length
        :param max_size: loads at most 'max_size' samples from the input file,
            if None loads the entire dataset
        """

        self.min_len = min_len
        self.max_len = max_len
        self.parallel = True
        self.sorted = False

        self.src = self.process_data(src_fname, tokenizer, max_size)
        self.tgt = self.process_data(tgt_fname, tokenizer, max_size)
        assert len(self.src) == len(self.tgt)

        self.filter_data(min_len, max_len)
        assert len(self.src) == len(self.tgt)

        src_lengths = [len(s) for s in self.src]
        tgt_lengths = [len(t) for t in self.tgt]
        self.src_lengths = torch.tensor(src_lengths)
        self.tgt_lengths = torch.tensor(tgt_lengths)
        self.lengths = self.src_lengths + self.tgt_lengths

        if sort:
            self.sort_by_length()

    def sort_by_length(self):
        """
        Sorts dataset by the sequence length.
        """
        self.lengths, indices = self.lengths.sort(descending=True)

        self.src = [self.src[idx] for idx in indices]
        self.tgt = [self.tgt[idx] for idx in indices]
        self.src_lengths = [self.src_lengths[idx] for idx in indices]
        self.tgt_lengths = [self.tgt_lengths[idx] for idx in indices]
        self.indices = indices.tolist()
        self.sorted = True

    def filter_data(self, min_len, max_len):
        """
        Preserves only samples which satisfy the following inequality:
            min_len <= src sample sequence length <= max_len AND
            min_len <= tgt sample sequence length <= max_len

        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        """
        logging.info(f'Filtering data, min len: {min_len}, max len: {max_len}')

        initial_len = len(self.src)
        filtered_src = []
        filtered_tgt = []
        for src, tgt in zip(self.src, self.tgt):
            if min_len <= len(src) <= max_len and \
                    min_len <= len(tgt) <= max_len:
                filtered_src.append(src)
                filtered_tgt.append(tgt)

        self.src = filtered_src
        self.tgt = filtered_tgt
        filtered_len = len(self.src)
        logging.info(f'Pairs before: {initial_len}, after: {filtered_len}')

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


class LazyParallelDataset(TextDataset):
    def __init__(self, src_fname, tgt_fname, tokenizer,
                 min_len, max_len, sort=False, max_size=None):
        """
        Constructor for the LazyParallelDataset.
        Tokenization is done on the fly.

        :param src_fname: path to the file with src language data
        :param tgt_fname: path to the file with tgt language data
        :param tokenizer: tokenizer
        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        :param sort: sorts dataset by sequence length
        :param max_size: loads at most 'max_size' samples from the input file,
            if None loads the entire dataset
        """
        self.min_len = min_len
        self.max_len = max_len
        self.parallel = True
        self.sorted = False
        self.tokenizer = tokenizer

        self.raw_src = self.process_raw_data(src_fname, max_size)
        self.raw_tgt = self.process_raw_data(tgt_fname, max_size)
        assert len(self.raw_src) == len(self.raw_tgt)

        logging.info(f'Filtering data, min len: {min_len}, max len: {max_len}')
        # Subtracting 2 because EOS and BOS are added later during tokenization
        self.filter_raw_data(min_len - 2, max_len - 2)
        assert len(self.raw_src) == len(self.raw_tgt)

        # Adding 2 because EOS and BOS are added later during tokenization
        src_lengths = [i + 2 for i in self.src_len]
        tgt_lengths = [i + 2 for i in self.tgt_len]
        self.src_lengths = torch.tensor(src_lengths)
        self.tgt_lengths = torch.tensor(tgt_lengths)
        self.lengths = self.src_lengths + self.tgt_lengths

    def process_raw_data(self, fname, max_size):
        """
        Loads data from the input file.

        :param fname: input file name
        :param max_size: loads at most 'max_size' samples from the input file,
            if None loads the entire dataset
        """
        logging.info(f'Processing data from {fname}')
        data = []
        with open(fname) as dfile:
            for idx, line in enumerate(dfile):
                if max_size and idx == max_size:
                    break
                data.append(line)
        return data

    def filter_raw_data(self, min_len, max_len):
        """
        Preserves only samples which satisfy the following inequality:
            min_len <= src sample sequence length <= max_len AND
            min_len <= tgt sample sequence length <= max_len

        :param min_len: minimum sequence length
        :param max_len: maximum sequence length
        """
        initial_len = len(self.raw_src)
        filtered_src = []
        filtered_tgt = []
        filtered_src_len = []
        filtered_tgt_len = []
        for src, tgt in zip(self.raw_src, self.raw_tgt):
            src_len = src.count(' ') + 1
            tgt_len = tgt.count(' ') + 1
            if min_len <= src_len <= max_len and \
                    min_len <= tgt_len <= max_len:
                filtered_src.append(src)
                filtered_tgt.append(tgt)
                filtered_src_len.append(src_len)
                filtered_tgt_len.append(tgt_len)

        self.raw_src = filtered_src
        self.raw_tgt = filtered_tgt
        self.src_len = filtered_src_len
        self.tgt_len = filtered_tgt_len
        filtered_len = len(self.raw_src)
        logging.info(f'Pairs before: {initial_len}, after: {filtered_len}')

    def __getitem__(self, idx):
        src = torch.tensor(self.tokenizer.segment(self.raw_src[idx]))
        tgt = torch.tensor(self.tokenizer.segment(self.raw_tgt[idx]))
        return src, tgt

    def __len__(self):
        return len(self.raw_src)
