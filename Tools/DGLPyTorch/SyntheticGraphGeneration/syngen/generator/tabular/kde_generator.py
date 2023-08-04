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

from tqdm import tqdm
from typing import Union, List, Optional
import pickle
import cupy as cp
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from cuml.neighbors import KernelDensity as KernelDensityGPU
from sklearn.preprocessing import OrdinalEncoder

from syngen.generator.tabular.chunked_tabular_generator import ChunkedBaseTabularGenerator

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class KDEGenerator(ChunkedBaseTabularGenerator):
    def __init__(self, **kwargs):
        """
            A tabular generator based on kernel density estimation.

            Categorical and continuous columns are modeled
            using gaussian KDE
        """
        super().__init__(**kwargs)

    def ordinal_encoder(self, cat_col):
        encoder = OrdinalEncoder()
        encoder.fit(cat_col)
        return encoder

    def fit(
            self,
            data: pd.DataFrame,
            categorical_columns: list = (),
            samples: Union[float, int] = 0.1,
            columns: Optional[List[str]] = None,
            verbose: bool = False,
    ):
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

        # - kde distribution
        cat_cols = tqdm(self.categorical_columns) if verbose else self.categorical_columns
        for column in cat_cols:
            col_data = data[column].dropna().values.reshape(-1, 1)
            enc = self.ordinal_encoder(col_data)
            col_data = enc.transform(col_data).reshape(-1, 1)
            kde = KernelDensity(kernel="gaussian")
            kde = kde.fit(col_data)
            self.cat_fit[column] = {
                "encoder": enc,
                "n_categories": len(enc.categories_[0]),
                "sampler": kde,
                'dtype': data[column].dtype,
            }
        self.cont_fit = {}
        # - gaussian distribution
        cont_cols = tqdm(self.continuous_columns) if verbose else self.continuous_columns
        for column in cont_cols:
            col_data = data[column].values.reshape(-1, 1)
            kde = KernelDensity(kernel="gaussian")
            kde = kde.fit(col_data)
            self.cont_fit[column] = {
                "sampler": kde,
                'dtype': data[column].dtype,
            }
        self.fits = {**self.cat_fit, **self.cont_fit}

    def sample(self, n, gpu=False, memmap_kwargs=None, start_idx=0, end_idx=None, **kwargs):

        use_memmap = memmap_kwargs is not None

        if use_memmap:
            memmap_outfile = np.load(memmap_kwargs['filename'], mmap_mode='r+')

        df = pd.DataFrame()
        if gpu:
            for column_id, column in enumerate(self.column_order):
                sampler = self.fits[column]["sampler"]

                gpu_sampler = KernelDensityGPU(kernel="gaussian")
                gpu_sampler.fit(np.asarray(sampler.tree_.data))

                if "encoder" in self.fits[column]:
                    # - must be categorical
                    encoder = self.fits[column]["encoder"]
                    n_categories = self.fits[column]["n_categories"]
                    sampled_data = gpu_sampler.sample(n)
                    sampled_data = cp.abs(sampled_data.reshape(-1, 1))
                    sampled_data = cp.round(sampled_data)
                    sampled_data = cp.clip(sampled_data, 0, n_categories - 1)
                    sampled_data = cp.asnumpy(sampled_data)
                    sampled_data = encoder.inverse_transform(sampled_data).reshape(-1)
                else:
                    sampled_data = gpu_sampler.sample(n)
                    sampled_data = cp.asnumpy(sampled_data.reshape(-1))
                sampled_data = sampled_data.astype(self.fits[column]["dtype"])
                if use_memmap:
                    memmap_outfile[start_idx:end_idx, column_id] = sampled_data
                else:
                    df[column] = sampled_data
        else:
            for column_id, column in enumerate(self.column_order):
                sampler = self.fits[column]["sampler"]
                if "encoder" in self.fits[column]:
                    # - must be categorical
                    encoder = self.fits[column]["encoder"]
                    n_categories = self.fits[column]["n_categories"]
                    sampled_data = sampler.sample(n)
                    sampled_data = np.abs(sampled_data.reshape(-1, 1))
                    sampled_data = np.round(sampled_data)
                    sampled_data = np.clip(sampled_data, 0, n_categories - 1)
                    sampled_data = encoder.inverse_transform(sampled_data).reshape(-1)
                else:
                    sampled_data = sampler.sample(n).reshape(-1)
                sampled_data = sampled_data.astype(self.fits[column]["dtype"])
                if use_memmap:
                    memmap_outfile[start_idx:end_idx, column_id] = sampled_data
                else:
                    df[column] = sampled_data

        if use_memmap:
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
