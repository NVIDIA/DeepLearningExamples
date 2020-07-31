# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch
from torch.utils.data import DataLoader

class PadDataLoader(DataLoader):

    @staticmethod
    def pad_collate_fn(batch):
        """
        Apply zero-padding.
        """
        # TODO refactor
        result = dict()
        for key in batch[0].keys():
            # apply padding on dataset
            sub_batch = [elem[key] for elem in batch]
            # check diff dims
            if not isinstance(sub_batch[0], np.ndarray):
                # if list of float or int
                assert all([type(x) == type(sub_batch[0]) for x in sub_batch[1:]]), sub_batch
                if isinstance(sub_batch[0], int):
                    sub_batch = torch.LongTensor(sub_batch)
                elif isinstance(sub_batch[0], float):
                    sub_batch = torch.DoubleTensor(sub_batch)

            elif any(list(map(lambda x: x.shape != sub_batch[0].shape, sub_batch[1:]))):
                sub_batch = torch.from_numpy(__class__.pad_zero(sub_batch))
            else:
                sub_batch = torch.from_numpy(np.concatenate(np.expand_dims(sub_batch, axis=0)))
            result[key] = sub_batch
        return result

    def __init__(self, dataset, batch_size, num_workers, shuffle=True, pin_memory=True, drop_last=True):
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         collate_fn=self.pad_collate_fn,
                         drop_last=drop_last
                         )

    @staticmethod
    def pad_zero(sub_batch):
        dims = [b.shape for b in sub_batch]

        max_dims = list(dims[0])
        for d_li in dims[1:]:
            for d_idx in range(len(d_li)):
                if max_dims[d_idx] < d_li[d_idx]:
                    max_dims[d_idx] = d_li[d_idx]

        temp = np.zeros((len(sub_batch), *max_dims), dtype=sub_batch[0].dtype)
        for i, b in enumerate(sub_batch):
            if len(b.shape) == 1:
                temp[i, :b.shape[0]] = b
            elif len(b.shape) == 2:
                temp[i, :b.shape[0], :b.shape[1]] = b
            elif len(b.shape) == 3:
                temp[i, :b.shape[0], :b.shape[1], :b.shape[2]] = b
            else:
                raise ValueError
        return temp


