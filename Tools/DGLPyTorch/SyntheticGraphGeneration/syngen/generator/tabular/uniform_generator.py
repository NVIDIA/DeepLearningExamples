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

from functools import partial
import pickle
from typing import Optional, List, Union
from tqdm import tqdm

import cupy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pandas.api.types import is_integer_dtype

from syngen.generator.tabular.chunked_tabular_generator import ChunkedBaseTabularGenerator


class UniformGenerator(ChunkedBaseTabularGenerator):
    """Uniform random feature generator.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def ordinal_encoder(self, cat_col):
        encoder = OrdinalEncoder()
        encoder.fit(cat_col)
        return encoder

    def fit(
            self,
            data,
            categorical_columns=(),
            samples: Union[float, int] = 0.1,
            columns: Optional[List[str]] = None,
            verbose: bool = False,
    ):
        """Computes the min and max ranges of the columns.

        Args:
            data: input data to use for extracting column statistics
            categorical_columns (list): list of columns that should be treated as categorical.
            verbose (bool): print intermediate results (default: False)
        """

        if samples > 0:
            num_samples = len(data)
            if 0.0 <= samples <= 1.0:
                num_samples = samples * num_samples
            else:
                num_samples = samples
            num_samples = min(int(num_samples), 10_000_000)
            data = data.sample(n=num_samples)

        self.column_order = columns or list(data.columns)
        self.cat_fit = {}
        self.categorical_columns = set(categorical_columns)
        self.continuous_columns = set(self.column_order) - self.categorical_columns

        cat_cols = tqdm(self.categorical_columns) if verbose else self.categorical_columns
        for column in cat_cols:
            enc = self.ordinal_encoder(data[column].values.reshape(-1, 1))
            n_unique = len(enc.categories_[0])
            self.cat_fit[column] = {
                "encoder": enc,
                "n_unique": n_unique,
                "sampler": partial(np.random.randint, 0, n_unique),
                'dtype': data[column].dtype,
            }

        self.cont_fit = {}
        self.integer_continuous_columns = []
        cont_cols = tqdm(self.continuous_columns) if verbose else self.continuous_columns
        for column in cont_cols:
            min_, max_ = data[column].min(), data[column].max()
            self.cont_fit[column] = {
                "min": min_,
                "max": max_,
                "sampler": partial(np.random.uniform, min_, max_),
                'dtype': data[column].dtype,
            }
            if is_integer_dtype(data[column].dtype):
                self.integer_continuous_columns.append(column)
        self.fits = {**self.cat_fit, **self.cont_fit}

    def sample(self, n, gpu=False, memmap_kwargs=None, start_idx=0, end_idx=None, **kwargs):

        use_memmap = memmap_kwargs is not None

        if use_memmap:
            memmap_outfile = np.load(memmap_kwargs['filename'], mmap_mode='r+')

        if gpu:
            cont_min = []
            cont_max = []

            for column in self.continuous_columns:
                cont_min.append(self.fits[column]['min'])
                cont_max.append(self.fits[column]['max'])

            cont_data = cp.random.uniform(
                cp.array(cont_min),
                cp.array(cont_max),
                size=(n, len(self.continuous_columns)),
                dtype=cp.float32
            )
            cont_data = cp.asnumpy(cont_data)
            df = pd.DataFrame(cont_data, columns=list(self.continuous_columns))
            if self.integer_continuous_columns:
                df[self.integer_continuous_columns] = \
                    df[self.integer_continuous_columns].astype(np.int32)

            for column in self.categorical_columns:
                sampled_data = cp.random.randint(0, self.fits[column]["n_unique"], size=n,  dtype=cp.int32)
                sampled_data = cp.asnumpy(sampled_data.reshape(-1, 1))
                encoder = self.fits[column]["encoder"]
                sampled_data = encoder.inverse_transform(sampled_data)
                df[column] = sampled_data.reshape(-1).astype(self.fits[column]["dtype"])

        else:
            df = pd.DataFrame()
            for column in self.column_order:
                sampler = self.fits[column]["sampler"]
                sampled_data = sampler(n)
                sampled_data = sampled_data.reshape(-1, 1)
                if "encoder" in self.fits[column]:
                    encoder = self.fits[column]["encoder"]
                    sampled_data = encoder.inverse_transform(sampled_data)
                df[column] = sampled_data.reshape(-1).astype(self.fits[column]["dtype"])

        df = df[self.column_order]

        if use_memmap:
            memmap_outfile[start_idx:end_idx] = df.values
            return None
        return df

    def _space_complexity_factor(self):
        return 2.5

    def save(self, path):
        with open(path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_handler:
            model = pickle.load(file_handler)
        return model
