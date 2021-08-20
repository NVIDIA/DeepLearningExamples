# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import torch.distributed as dist
from abc import ABC
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from se3_transformer.runtime.utils import get_local_rank


def _get_dataloader(dataset: Dataset, shuffle: bool, **kwargs) -> DataLoader:
    # Classic or distributed dataloader depending on the context
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    return DataLoader(dataset, shuffle=(shuffle and sampler is None), sampler=sampler, **kwargs)


class DataModule(ABC):
    """ Abstract DataModule. Children must define self.ds_{train | val | test}. """

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        if get_local_rank() == 0:
            self.prepare_data()

        # Wait until rank zero has prepared the data (download, preprocessing, ...)
        if dist.is_initialized():
            dist.barrier(device_ids=[get_local_rank()])

        self.dataloader_kwargs = {'pin_memory': True, 'persistent_workers': True, **dataloader_kwargs}
        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self):
        """ Method called only once per node. Put here any downloading or preprocessing """
        pass

    def train_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_test, shuffle=False, **self.dataloader_kwargs)
