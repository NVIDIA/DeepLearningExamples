import logging

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader

import seq2seq.data.config as config
from seq2seq.data.sampler import BucketingSampler


def build_collate_fn(batch_first=False, parallel=True, sort=False):
    """
    Factor for collate_fn functions.

    :param batch_first: if True returns batches in (batch, seq) format, if not
        returns in (seq, batch) format
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
            key = lambda item: len(item[1])
            indices, src_seqs = zip(*sorted(enumerate(src_seqs), key=key,
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
            key = lambda item: len(item[1])
            indices, src_seqs = zip(*sorted(enumerate(src_seqs), key=key,
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

        self.min_len = min_len
        self.max_len = max_len
        self.parallel = False

        self.src = self.process_data(src_fname, tokenizer, max_size)

        if min_len is not None and max_len is not None:
            self.filter_data(min_len, max_len)

        lengths = [len(s) for s in self.src]
        self.lengths = torch.tensor(lengths)

        if sort:
            self.sort_by_length()

    def sort_by_length(self):
        self.lengths, indices = self.lengths.sort(descending=True)

        self.src = [self.src[idx] for idx in indices]

    def filter_data(self, min_len, max_len):
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

    def get_loader(self, batch_size=1, shuffle=False, num_workers=0,
                   batch_first=False, drop_last=False, bucketing=True):

        collate_fn = build_collate_fn(batch_first, parallel=self.parallel,
                                      sort=True)

        if shuffle:
            sampler = BucketingSampler(self, batch_size, bucketing)
        else:
            sampler = SequentialSampler(self)

        return DataLoader(self,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          sampler=sampler,
                          num_workers=num_workers,
                          pin_memory=False,
                          drop_last=drop_last)


class ParallelDataset(TextDataset):
    def __init__(self, src_fname, tgt_fname, tokenizer,
                 min_len, max_len, sort=False, max_size=None):

        self.min_len = min_len
        self.max_len = max_len
        self.parallel = True

        self.src = self.process_data(src_fname, tokenizer, max_size)
        self.tgt = self.process_data(tgt_fname, tokenizer, max_size)
        assert len(self.src) == len(self.tgt)

        self.filter_data(min_len, max_len)
        assert len(self.src) == len(self.tgt)

        lengths = [len(s) + len(t) for (s, t) in zip(self.src, self.tgt)]
        self.lengths = torch.tensor(lengths)

        if sort:
            self.sort_by_length()

    def sort_by_length(self):
        self.lengths, indices = self.lengths.sort(descending=True)

        self.src = [self.src[idx] for idx in indices]
        self.tgt = [self.tgt[idx] for idx in indices]

    def filter_data(self, min_len, max_len):
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
