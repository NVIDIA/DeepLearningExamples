# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import os
import time
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset

class CriteoBinDataset(Dataset):
    """Simple dataloader for a recommender system. Designed to work with a single binary file."""

    def __init__(self, data_file, batch_size=1, subset=None,
                 numerical_features=13, categorical_features=26,
                 data_type='int32', online_shuffle=True):
        self.data_type = np.__dict__[data_type]
        bytes_per_feature = self.data_type().nbytes

        self.tad_fea = 1 + numerical_features
        self.tot_fea = 1 + numerical_features + categorical_features

        self.batch_size = batch_size
        self.bytes_per_entry = (bytes_per_feature * self.tot_fea * batch_size)

        self.num_entries = math.ceil(os.path.getsize(data_file) / self.bytes_per_entry)

        if subset is not None:
            if subset <= 0 or subset > 1:
                raise ValueError('Subset parameter must be in (0,1) range')
            self.num_entries = self.num_entries * subset

        print('data file:', data_file, 'number of batches:', self.num_entries)
        self.file = open(data_file, 'rb')
        self.online_shuffle=online_shuffle

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        if idx == 0:
            self.file.seek(0, 0)

        if self.online_shuffle:
            self.file.seek(idx * self.bytes_per_entry, 0)

        raw_data = self.file.read(self.bytes_per_entry)
        array = np.frombuffer(raw_data, dtype=self.data_type).reshape(-1, self.tot_fea)

        # numerical features are encoded as float32
        numerical_features = array[:, 1:self.tad_fea].view(dtype=np.float32)
        numerical_features = torch.from_numpy(numerical_features)


        categorical_features = torch.from_numpy(array[:, self.tad_fea:])
        labels = torch.from_numpy(array[:, 0])

        return numerical_features, categorical_features, labels

    def __del__(self):
        self.file.close()


if __name__ == '__main__':
    print('Dataloader benchmark')

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()

    dataset = CriteoBinDataset(data_file=args.file, batch_size=args.batch_size)

    begin = time.time()
    for i in range(args.steps):
        _ = dataset[i]
    end = time.time()
    
    step_time = (end - begin) / args.steps
    throughput = args.batch_size / step_time

    print(f'Mean step time: {step_time:.6f} [s]')
    print(f'Mean throughput: {throughput:,.0f} [samples / s]')
