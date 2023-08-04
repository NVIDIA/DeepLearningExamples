# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

from typing import Optional, List

import cupy as cp
import pickle
import numpy as np
import pandas as pd


from syngen.generator.tabular.chunked_tabular_generator import ChunkedBaseTabularGenerator

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class RandomMVGenerator(ChunkedBaseTabularGenerator):
    """Random Multivariate Gaussian generator
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ndims = None
        self.column_order = None

    def fit(self, data: Optional[pd.DataFrame] = None, ndims: Optional[int] = None,
            columns: Optional[List[str]] = None,
            categorical_columns=(), verbose=False):
        """
            random ignores categorical columns at the moment

        """

        assert ndims is not None or data is not None or self.ndims is not None or columns is not None

        if data is not None:
            ndims = len(data.columns)
            self.column_order = list(data.columns)

        if columns is not None:
            self.column_order = columns
            ndims = len(columns)

        if ndims is None:
            ndims = self.ndims

        self.mu = np.random.randn(ndims).astype(np.float32)
        self.cov = np.eye(ndims) * np.abs(
            np.random.randn(ndims).reshape(-1, 1)
        ).astype(np.float32)
        self.ndims = ndims

    def _space_complexity_factor(self):
        return 2.0

    def sample(self, n, gpu=False, memmap_kwargs=None, start_idx=0, end_idx=None, **kwargs):

        use_memmap = memmap_kwargs is not None

        if use_memmap:
            memmap_outfile = np.load(memmap_kwargs['filename'], mmap_mode='r+')

        if gpu:
            samples = cp.random.multivariate_normal(self.mu, self.cov, size=n, dtype=cp.float32)
            samples = cp.asnumpy(samples)
        else:
            samples = np.random.multivariate_normal(self.mu, self.cov, size=n).astype(np.float32)

        if use_memmap:
            memmap_outfile[start_idx:end_idx] = samples
            return None
        else:
            df = pd.DataFrame(samples)
            if self.column_order is None:
                df.columns = df.columns.astype(str)
            else:
                df.columns = self.column_order
            return df

    def save(self, path):
        with open(path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_handler:
            model = pickle.load(file_handler)
        return model
