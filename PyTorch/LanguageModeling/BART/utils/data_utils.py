# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

import numpy as np
import torch

try:
    from .utils import LegacySeq2SeqDataset
except ImportError:
    from utils.utils import LegacySeq2SeqDataset


from torch.utils.data import DataLoader
import distributed_utils

class Seq2SeqDataLoader(DataLoader):
    def __init__(self, type_path, data_dir, tokenizer, batch_size, device='cpu',
        max_source_length=1024, max_target_length=1024, n_obs=None,
        shuffle=False, sortish_sampler=False, num_workers=4):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.sortish_sampler = sortish_sampler

        self.device = device
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.dataset = self.get_dataset(type_path)
        # Partition data for DistributedDataParallel
        world_size = distributed_utils.get_world_size()
        rank = distributed_utils.get_rank()

        sampler = None
        if world_size > 1 and type_path == "train":
            sampler =self.dataset.make_sortish_sampler(batch_size, distributed=True, rank=rank, num_replicas=world_size)
            shuffle = False

        super().__init__(
            self.dataset,
            batch_size=batch_size,
            collate_fn=self.dataset.collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
        )

    def get_dataset(self, type_path):
        dataset = LegacySeq2SeqDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            type_path=type_path,
            n_obs=self.n_obs,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            src_lang="", tgt_lang=""
        )
        return dataset
