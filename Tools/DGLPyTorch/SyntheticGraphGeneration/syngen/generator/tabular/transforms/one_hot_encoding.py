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

import numpy as np
import pandas as pd

from syngen.generator.tabular.transforms.base_transform import BaseTransform


class OneHotEncoding(BaseTransform):
    """OneHotEncoding for categorical data.
    Adopted from: https://github.com/sdv-dev/CTGAN

    This transformer replaces a single vector with N unique categories in it
    with N vectors which have 1s on the rows where the corresponding category
    is found and 0s on the rest.

    Null values are considered just another category.

    Args:
        error_on_unknown (bool):
            If a value that was not seen during the fit stage is passed to
            transform, then an error will be raised if this is True.
    """

    dummies = None
    _dummy_na = None
    _num_dummies = None
    _dummy_encoded = False
    _indexer = None
    _uniques = None

    def __init__(self, error_on_unknown=True):
        self.error_on_unknown = error_on_unknown

    @staticmethod
    def _prepare_data(data):
        """Convert data to appropriate format.

        If data is a valid list or a list of lists,
        transforms it into an np.array, otherwise returns it.

        Args:
            data (pandas.Series, numpy.ndarray, list or list of lists):
                Data to prepare.

        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError("Unexpected format.")
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError("Unexpected format.")

            data = data[:, 0]

        return data

    def _transform(self, data):
        if self._dummy_encoded:
            coder = self._indexer
            codes = pd.Categorical(data, categories=self._uniques).codes
        else:
            coder = self._uniques
            codes = data

        rows = len(data)
        dummies = np.broadcast_to(coder, (rows, self._num_dummies))
        coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
        array = (coded == dummies).astype(int)

        if self._dummy_na:
            null = np.zeros((rows, 1), dtype=int)
            null[pd.isnull(data)] = 1
            array = np.append(array, null, axis=1)

        return array

    def fit(self, data):
        """Fit the transformer to the data.

        Get the pandas `dummies` which will be used later on for OneHotEncoding.

        Args:
            data (pandas.Series, numpy.ndarray, list or list of lists):
                Data to fit the transformer to.
        """
        data = self._prepare_data(data)

        null = pd.isnull(data)
        self._uniques = list(pd.unique(data[~null]))
        self._dummy_na = null.any()
        self._num_dummies = len(self._uniques)
        self._indexer = list(range(self._num_dummies))
        self.dummies = self._uniques.copy()

        if not np.issubdtype(data.dtype, np.number):
            self._dummy_encoded = True

        if self._dummy_na:
            self.dummies.append(np.nan)

    def transform(self, data):
        """Replace each category with the OneHot vectors.

        Args:
            data (pandas.Series, numpy.ndarray, list or list of lists):
                Data to transform.

        Returns:
            numpy.ndarray:
        """
        data = self._prepare_data(data)
        array = self._transform(data)

        if self.error_on_unknown:
            unknown = array.sum(axis=1) == 0
            if unknown.any():
                raise ValueError(
                    f"Attempted to transform {list(data[unknown])} ",
                    "that were not seen during fit stage.",
                )

        return array

    def inverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)
        return pd.Series(indices).map(self.dummies.__getitem__)
