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

from abc import ABC


class BaseDataTransformer(ABC):
    """Base class for all data transformers.
    The `BaseDataTransformer` provides the transformation required by
    generators to transform (encode) and inverse_transform (decode) data.
    It contains the `fit`, `transform`, `inverse_transform`,
    and `get_metadata` functions that must be implemented by specific data
    transformer objects.
    """

    def fit(self, data):
        """Fits the data transform to the data. This is optional

        Args:
            data (pandas.Series or cudf.Series or numpy.array or cupy.array):
            Data to transform.

        Returns:
            None

        """
        pass

    def transform(self, data):
        """Transform the data.

        Args:
            data (pandas.Series or cudf.Series or numpy.array or cupy.array):
            Data to transform.

        Returns:
            numpy.array: Transformed data.
        """
        raise NotImplementedError()

    def fit_transform(self, data):
        """Fit to the data and then return the transformed data.

        Args:
            data (pandas.Series or cudf.Series or numpy.array or cupy.array):
            Data to fit and transform

        Returns:
            Transformed data.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """Reverses the transformation done on the data back to original values.

        Args:
            data (pandas.Series or cudf.Series or numpy.array or cupy.array):
            Data to inverse-transform.
        Returns:
            raw_data: inverse transformed data

        """
        raise NotImplementedError()
