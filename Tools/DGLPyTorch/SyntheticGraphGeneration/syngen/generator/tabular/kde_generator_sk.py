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

from collections import OrderedDict
from functools import partial
from typing import Union
import pickle
import numpy as np
import pandas as pd
import scipy
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OrdinalEncoder

from syngen.generator.tabular.base_tabular_generator import BaseTabularGenerator


class KDEGeneratorSK(BaseTabularGenerator):
    def __init__(self, **kwargs):
        """
            A tabular generator based on kernel density estimation
            the default.

            Categorical and continuous columns are modeled
            using gaussian KDE
        """
        super(BaseTabularGenerator).__init__()

    def ordinal_encoder(self, cat_col):
        encoder = OrdinalEncoder()
        encoder.fit(cat_col)
        return encoder

    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: list = [],
        samples: Union[float, int] = 0.1,
    ):

        num_samples = len(data)

        # - naive sampling
        if samples >= 0.0 and samples <= 1.0:
            num_samples = samples * num_samples
        else:
            num_samples = samples
        num_samples = int(num_samples)
        data = data.sample(n=num_samples)
        self.column_order = list(data.columns)
        self.cat_fit = {}
        self.categorical_columns = set(categorical_columns)
        self.continuous_columns = set(data.columns) - self.categorical_columns

        # - kde distribution
        for column in self.categorical_columns:
            enc = self.ordinal_encoder(data[column].values.reshape(-1, 1))
            col_data = data[column].values.reshape(-1, 1)
            col_data = enc.transform(col_data).reshape(-1, 1)
            kde = KernelDensity(kernel="gaussian")
            kde = kde.fit(col_data)
            self.cat_fit[column] = {
                "encoder": enc,
                "n_categories": len(enc.categories_[0]),
                "sampler": kde,
            }
        self.cont_fit = {}
        # - gaussian distribution
        for column in self.continuous_columns:
            col_data = data[column].values.reshape(-1, 1)
            kde = KernelDensity(kernel="gaussian")
            kde = kde.fit(col_data)
            self.cont_fit[column] = {"sampler": kde}

        self.fits = {**self.cat_fit, **self.cont_fit}
    
    def sample(self, n, **kwargs):
        data_dict = OrderedDict()
        for column in self.column_order:
            sampler = self.fits[column]["sampler"]
            if "encoder" in self.fits[column]:
                # - must be categorical
                encoder = self.fits[column]["encoder"]
                n_categories = self.fits[column]["n_categories"]
                sampled_data = sampler.sample(n)
                sampled_data = np.abs(sampled_data.reshape(-1, 1))
                sampled_data = np.round(sampled_data)
                sampled_data = np.clip(sampled_data, 0, n_categories - 1)
                sampled_data = encoder.inverse_transform(sampled_data)
            else:
                sampled_data = sampler.sample(n)
            data_dict[column] = list(sampled_data.reshape(-1))
        df = pd.DataFrame.from_dict(data_dict)
        return df

    def set_device(self, device):
        ...

    def save(self, path):
        with open(path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_handler:
            model = pickle.load(file_handler)
        return model
