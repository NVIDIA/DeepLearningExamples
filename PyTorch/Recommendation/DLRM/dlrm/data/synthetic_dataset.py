# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

import torch
import math
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """Synthetic dataset version of criteo dataset."""

    def __init__(self,  num_entries, device='cuda', batch_size=1, dense_features=13,
                 categorical_feature_sizes=None):
        # dataset. single target, 13 dense features, 26 sparse features
        self.sparse_features = len(categorical_feature_sizes)
        self.dense_features = dense_features

        self.tot_fea = 1 + dense_features + self.sparse_features
        self.batch_size = batch_size
        self.batches_per_epoch = math.ceil(num_entries / batch_size)
        self.categorical_feature_sizes = categorical_feature_sizes
        self.device = device

        self.tensor = torch.randint(low=0, high=2, size=(self.batch_size, self.tot_fea), device=self.device)
        self.tensor = self.tensor.float()

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        return self.tensor[:, 1:14], self.tensor[:, 14:], self.tensor[:, 0]
