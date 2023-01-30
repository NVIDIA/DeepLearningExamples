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

import cupy
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class RandomMVGenerator:
    """Random Multivariate Gaussian generator
    """

    def __init__(self, **kwargs):
        pass

    def fit(self, ndims):
        self.mu = np.random.randn(ndims)
        self.cov = np.eye(ndims) * np.abs(
            np.random.randn(ndims).reshape(-1, 1)
        )

    def sample(self, n):
        samples = cupy.random.multivariate_normal(self.mu, self.cov, size=n)
        samples = cupy.asnumpy(samples)
        df = pd.DataFrame(samples)
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
