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

import pandas
import dask.dataframe as dask_dataframe

_DFREADER = pandas
_DASKREADER = dask_dataframe

try:
    import cudf

    _DFREADER = cudf
except ImportError:
    pass

try:
    import dask_cudf

    _DASKREADER = dask_cudf
except ImportError:
    pass

READERS = ['pandas', 'dask', 'cudf', 'dask_cudf']

class DFReader(object):
    df_reader = _DFREADER
    dask_reader = _DASKREADER

    @staticmethod
    def get_df_reader():
        return DFReader.df_reader

    @staticmethod
    def get_df_reader_cpu():
        return pandas

    @staticmethod
    def get_dask_reader_cpu():
        return dask_dataframe

    @staticmethod
    def get_dask_reader():
        return DFReader.dask_reader
