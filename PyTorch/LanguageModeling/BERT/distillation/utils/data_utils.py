# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import six
import glob
import h5py
import torch
import random
import collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import utils
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler, Dataset

# model inputs - it's a bit nicer to use a namedtuple rather than keep the
# features as a dict
Inputs = collections.namedtuple(
    "Inputs", ["input_ids", "input_mask", "segment_ids", "masked_lm_positions",
               "masked_lm_ids", "masked_lm_weights"])


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)

class PretrainDataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]


class DatasetIterator:
    def __init__(self, config, batch_size, world_size=1, rank=0):
        self.config = config
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.index = 0
        self.future_dataloader = None

        self.worker_init = WorkerInitObj(config.seed + rank)
        self.pool = ProcessPoolExecutor(max_workers=1)
        self.num_files = len(config.input_files)

        # Bootstrap files if few than processes
        if self.num_files < world_size:
            lcm = (len(input_files) * world_size) // math.gcd(len(input_files), world_size)
            factor = lcm // self.num_files
            temp_input_files = []
            for i in range(factor):
                temp_input_files.extend(config.input_files)
            config.input_files = temp_input_files

        self.input_files = config.input_files[rank::world_size]

        random.seed(config.seed)
        random.shuffle(self.input_files)

    def __iter__(self):
        self.load_future()
        return self

    def __next__(self):
        dataloader = self.future_dataloader.result(timeout=None)
        self.index += 1
        if self.index >= len(self.input_files):
            self.index = 0
            random.shuffle(self.input_files)
        self.load_future()
        return dataloader

    def load_future(self):
        self.future_dataloader = self.pool.submit(
            create_dataset,
            self.input_files[self.index],
            self.config.max_seq_length,
            self.batch_size,
            self.worker_init
        )

    def load_state_dict(self, state_dict):
        self.index = state_dict['file_index']

    def state_dict(self):
        return {
            'file_index': self.index - 1, # We want to point to the current dataloader, not a future one
        }

def create_dataset(input_file, max_seq_length, batch_size, worker_init, num_cpu_threads=4):
    print("using file", input_file)
    dataset = PretrainDataset(
        input_file=input_file, max_pred_length=max_seq_length)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_cpu_threads,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=True)
    return dataloader
