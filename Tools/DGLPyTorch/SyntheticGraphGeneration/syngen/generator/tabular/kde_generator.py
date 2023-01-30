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

import abc
from collections import OrderedDict
from functools import partial
from typing import Union
import pickle
import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch import nn

from syngen.generator.tabular.base_tabular_generator import BaseTabularGenerator


class Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=0.05):
        """Initializes a new Kernel.
        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _diffs(self, test_data, train_data):
        """Computes difference between each x in test_data with all train_data."""
        test_data = test_data.view(test_data.shape[0], 1, *test_data.shape[1:])
        train_data = train_data.view(
            1, train_data.shape[0], *train_data.shape[1:]
        )
        return test_data - train_data

    @abc.abstractmethod
    def forward(self, test_data, train_data):
        """Computes p(x) for each x in test_data given train_data."""

    @abc.abstractmethod
    def sample(self, train_data):
        """Generates samples from the kernel distribution."""


class ParzenWindowKernel(Kernel):
    """Implementation of the Parzen window kernel."""

    def forward(self, test_data, train_data):
        abs_diffs = torch.abs(self._diffs(test_data, train_data))
        dims = tuple(range(len(abs_diffs.shape))[2:])
        dim = np.prod(abs_diffs.shape[2:])
        inside = torch.sum(abs_diffs / self.bandwidth <= 0.5, dim=dims) == dim
        coef = 1 / self.bandwidth ** dim
        return (coef * inside).mean(dim=1)

    def sample(self, train_data):
        device = train_data.device
        noise = (
            torch.rand(train_data.shape, device=device) - 0.5
        ) * self.bandwidth
        return train_data + noise


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_data, train_data):
        diffs = self._diffs(test_data, train_data)
        dims = tuple(range(len(diffs.shape))[2:])
        var = self.bandwidth ** 2
        exp = torch.exp(-torch.norm(diffs, p=2, dim=dims) ** 2 / (2 * var))
        coef = 1 / torch.sqrt(torch.tensor(2 * np.pi * var))
        return (coef * exp).mean(dim=1)

    def sample(self, train_data):
        device = train_data.device
        noise = torch.randn(train_data.shape, device=device) * self.bandwidth
        return train_data + noise


class KernelDensity(abc.ABC):
    def __init__(self, kernel="gaussian"):
        """Initializes a new KernelDensity.

        Args:
            kernel (str): The kernel to place on each of the data points.
        """
        super().__init__()

        if kernel == "gaussian":
            self.kernel = GaussianKernel()
        elif kernel == "parzen":
            self.kernel = ParzenWindowKernel()

    def fit(self, data):
        self.train_data = data

    @property
    def device(self):
        return self.train_data.device

    def forward(self, x):
        return self.kernel(x, self.train_data)

    def sample(self, n_samples):
        idxs = np.random.choice(range(len(self.train_data)), size=n_samples)
        return self.kernel.sample(self.train_data[idxs])


class KDEGenerator(BaseTabularGenerator):
    def __init__(self, device="cuda", **kwargs):
        """
            A feature generator based on kernel density estimation
            the default.

            Categorical and continuous columns are modeled using gaussian KDE

            Args:
                device (str): device to use (default: cuda)
        """
        super(BaseTabularGenerator).__init__()
        self.device = device

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
            col_data = torch.from_numpy(col_data).to(device=self.device)
            kde = KernelDensity(kernel="gaussian")
            kde.fit(col_data)
            self.cat_fit[column] = {
                "encoder": enc,
                "n_categories": len(enc.categories_[0]),
                "sampler": kde,
            }
        self.cont_fit = {}
        # - gaussian distribution
        for column in self.continuous_columns:
            col_data = data[column].values.reshape(-1, 1)
            col_data = torch.from_numpy(col_data).to(device=self.device)
            kde = KernelDensity(kernel="gaussian")
            kde.fit(col_data)
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
                if sampled_data.is_cuda:
                    sampled_data = sampled_data.cpu().numpy()
                else:
                    sampled_data = sampled_data.numpy()

                sampled_data = np.abs(sampled_data.reshape(-1, 1))
                sampled_data = np.round(sampled_data)
                sampled_data = np.clip(sampled_data, 0, n_categories - 1)
                sampled_data = encoder.inverse_transform(sampled_data)
            else:
                sampled_data = sampler.sample(n)

                if sampled_data.is_cuda:
                    sampled_data = sampled_data.cpu().numpy()
                else:
                    sampled_data = sampled_data.numpy()

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
