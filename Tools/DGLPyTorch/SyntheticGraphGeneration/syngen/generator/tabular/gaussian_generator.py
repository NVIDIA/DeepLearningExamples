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
import scipy
from sklearn.preprocessing import OrdinalEncoder

from syngen.generator.tabular.base_tabular_generator import BaseTabularGenerator


class GaussianGenerator(BaseTabularGenerator):
    def __init__(self, **kwargs):
        super(BaseTabularGenerator).__init__()

    def ordinal_encoder(self, cat_col):
        encoder = OrdinalEncoder()
        encoder.fit(cat_col)
        return encoder

    def fit(self, data, categorical_columns=()):
        self.column_order = list(data.columns)
        self.cat_fit = {}
        self.categorical_columns = set(categorical_columns)
        self.continuous_columns = set(data.columns) - self.categorical_columns
        num_samples = len(data)

        # - multinomial distribution
        for column in self.categorical_columns:
            enc = self.ordinal_encoder(data[column].values.reshape(-1, 1))
            pvals = data[column].value_counts() / num_samples
            pvals = pvals.values
            self.cat_fit[column] = {
                "encoder": enc,
                "pvals": pvals,
            }

        self.cont_fit = {}
        # - gaussian distribution
        for column in self.continuous_columns:
            mean, std = data[column].mean(), data[column].std()
            self.cont_fit[column] = {
                "mean": mean,
                "std": std,
            }
        self.fits = {**self.cat_fit, **self.cont_fit}

    def sample(self, n, **kwargs):
        data_dict = OrderedDict()
        for column in self.column_order:
            if column in self.categorical_columns:
                sampled_data = np.random.multinomial(n,
                    self.fits[column]["pvals"])
                fsd = []
                for i, sd in enumerate(sampled_data):
                    t = [i] * sd
                    if len(t):
                        fsd += t

                sampled_data = np.asarray(fsd, dtype=np.int32)
                np.random.shuffle(sampled_data)
                sampled_data = sampled_data.reshape(-1, 1)
                encoder = self.fits[column]["encoder"]
                sampled_data = encoder.inverse_transform(sampled_data)
            else:
                sampled_data = np.random.normal(
                        self.fits[column]['mean'],
                        self.fits[column]['std'], n)

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
