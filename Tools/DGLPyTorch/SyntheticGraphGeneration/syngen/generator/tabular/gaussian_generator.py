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

import pickle
from typing import Optional, List

import cupy as cp

import numpy as np
import pandas as pd

from tqdm import tqdm
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import OrdinalEncoder

from syngen.generator.tabular.chunked_tabular_generator import ChunkedBaseTabularGenerator
from syngen.generator.utils import cuda_repeat


class GaussianGenerator(ChunkedBaseTabularGenerator):
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
            columns: Optional[List[str]] = None,
            verbose: bool = False,
    ):
        self.column_order = columns or list(data.columns)
        self.cat_fit = {}
        self.categorical_columns = set(categorical_columns)
        self.continuous_columns = set(self.column_order) - self.categorical_columns
        num_samples = len(data)

        # - multinomial distribution
        cat_cols = tqdm(self.categorical_columns) if verbose else self.categorical_columns
        for column in cat_cols:
            enc = self.ordinal_encoder(data[column].values.reshape(-1, 1))
            pvals = data[column].value_counts() / num_samples
            pvals = pvals.values
            self.cat_fit[column] = {
                "encoder": enc,
                "pvals": pvals,
                'dtype': data[column].dtype,
            }

        self.cont_fit = {}
        self.integer_continuous_columns = []
        # - gaussian distribution
        cont_cols = tqdm(self.continuous_columns) if verbose else self.continuous_columns
        for column in cont_cols:
            mean, std = data[column].mean(), data[column].std()
            self.cont_fit[column] = {
                "mean": mean,
                "std": std,
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
            cont_means = []
            cont_stds = []

            for column in self.continuous_columns:
                cont_means.append(self.fits[column]['mean'])
                cont_stds.append(self.fits[column]['std'])

            cont_data = cp.random.normal(
                cp.array(cont_means),
                cp.array(cont_stds),
                size=(n, len(self.continuous_columns)),
                dtype=cp.float32
            )
            cont_data = cp.asnumpy(cont_data)
            df = pd.DataFrame(cont_data, columns=list(self.continuous_columns))
            if self.integer_continuous_columns:
                df[self.integer_continuous_columns] = \
                    df[self.integer_continuous_columns].astype(np.int32)

            for column in self.categorical_columns:
                sampled_data = cp.random.multinomial(n, self.fits[column]["pvals"])
                sampled_data = cuda_repeat(sampled_data)
                cp.random.shuffle(sampled_data)
                sampled_data = cp.asnumpy(sampled_data.reshape(-1, 1))
                encoder = self.fits[column]["encoder"]
                sampled_data = encoder.inverse_transform(sampled_data)
                df[column] = sampled_data.reshape(-1).astype(self.fits[column]["dtype"])
        else:
            df = pd.DataFrame()
            for column in self.column_order:
                if column in self.categorical_columns:
                    sampled_data = np.random.multinomial(n,
                        self.fits[column]["pvals"])
                    sampled_data = np.repeat(np.arange(len(sampled_data)), sampled_data)
                    np.random.shuffle(sampled_data)
                    sampled_data = sampled_data.reshape(-1, 1)
                    encoder = self.fits[column]["encoder"]
                    sampled_data = encoder.inverse_transform(sampled_data)
                else:
                    sampled_data = np.random.normal(
                            self.fits[column]['mean'],
                            self.fits[column]['std'], n)
                df[column] = sampled_data.reshape(-1).astype(self.fits[column]["dtype"])

        df = df[self.column_order]

        if use_memmap:
            memmap_outfile[start_idx:end_idx] = df.values
            return None
        return df

    def save(self, path):
        with open(path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_handler:
            model = pickle.load(file_handler)
        return model
