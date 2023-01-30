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
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from syngen.generator.tabular.base_tabular_generator import BaseTabularGenerator


class UniformGenerator(BaseTabularGenerator):
    """Uniform random feature generator.
    """

    def __init__(self, **kwargs):
        super(BaseTabularGenerator).__init__()

    def ordinal_encoder(self, cat_col):
        encoder = OrdinalEncoder()
        encoder.fit(cat_col)
        return encoder

    def fit(self, data, categorical_columns=()):
        """Computes the min and max ranges of the columns.

        Args:
            data: input data to use for extracting column statistics
            categorical_columns (list): list of columns that should be treated as categorical.
        """
        self.column_order = list(data.columns)
        self.cat_fit = {}
        self.categorical_columns = set(categorical_columns)
        self.continuous_columns = set(data.columns) - self.categorical_columns

        for column in self.categorical_columns:
            enc = self.ordinal_encoder(data[column].values.reshape(-1, 1))
            n_unique = len(enc.categories_[0])
            self.cat_fit[column] = {
                "encoder": enc,
                "sampler": partial(np.random.randint, 0, n_unique),
            }

        self.cont_fit = {}
        for column in self.continuous_columns:
            min_, max_ = data[column].min(), data[column].max()
            self.cont_fit[column] = {
                "min": min_,
                "max": max_,
                "sampler": partial(np.random.uniform, min_, max_),
            }
        self.fits = {**self.cat_fit, **self.cont_fit}

    def sample(self, n, **kwargs):
        data_dict = OrderedDict()
        for column in self.column_order:
            sampler = self.fits[column]["sampler"]
            sampled_data = sampler(n)
            sampled_data = sampled_data.reshape(-1, 1)

            if "encoder" in self.fits[column]:
                encoder = self.fits[column]["encoder"]
                sampled_data = encoder.inverse_transform(sampled_data)

            data_dict[column] = list(sampled_data.reshape(-1))
        df = pd.DataFrame.from_dict(data_dict)
        return df

    def save(self, path):
        with open(path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_handler:
            model = pickle.load(file_handler)
        return model

    def set_device(self, device):
        ...