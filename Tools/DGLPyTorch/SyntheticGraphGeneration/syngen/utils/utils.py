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

import time
import logging
import importlib
from typing import Optional

import cudf
import cupy
import dask_cudf
import pandas as pd

from syngen.utils.types import DataFrameType

logger = logging.getLogger(__name__)
log = logger


class CustomTimer:
    """Wraps `time` module and adds tagging for multiple timers

    Example:
        timer = CustomTimer()
        timer.start_counter("tag")
        # - do a series of operation
        # ...
        # - end of operations
        timer.end_counter("tag", "tag timer has ended")

    Args:
        path (Optional[str])

    """

    def __init__(self, path: Optional[str] = str):
        self.path = path
        self.timers = {}
        self.f = None
        if self.path:
            self.f = open(self.path, "w")

    def start_counter(self, key: str):
        self.timers[key] = time.perf_counter()

    def end_counter(self, key: str, msg: str):
        end = time.perf_counter()
        start = self.timers.get(key)
        if self.f and start:
            self.f.write(f"{msg}: {end - start:.2f}\n")

    def maybe_close(self):
        if self.f:
            self.f.close()


def current_ms_time():
    return round(time.time() * 1000)


def df_to_pandas(df):
    """ Converts `DataFrameType` to `pandas.DataFrame`

        Args:
            df (DataFrameType): the DataFrame to be converted
    """
    if isinstance(df, cudf.DataFrame):
        pddf = df.to_pandas()
    elif isinstance(df, dask_cudf.DataFrame):
        pddf = pd.DataFrame(
            cupy.asnumpy(df.values.compute()), columns=df.columns
        )
    elif isinstance(df, pd.DataFrame):
        pddf = df
    else:
        raise ValueError(f"DataFrame type {type(df)} not supported")
    return pddf


def df_to_cudf(df: DataFrameType):
    """ Converts `DataFrameType` to `cudf.DataFrame`

        Args:
            df (DataFrameType): the DataFrame to be converted
    """
    if isinstance(df, cudf.DataFrame):
        pass
    elif isinstance(df, dask_cudf.DataFrame):
        df = cudf.DataFrame(
            cupy.asnumpy(df.values.compute()), columns=df.columns
        )
    elif isinstance(df, pd.DataFrame):
        df = cudf.from_pandas(df)
    else:
        raise ValueError(f"DataFrameType type {type(df)} not supported")
    return df


def df_to_dask_cudf(df: DataFrameType,
                    chunksize: Optional[int]=None):
    """ Converts `DataFrameType` to `dask_cudf.DataFrame`

        Args:
            df (DataFrameType): the DataFrame to be converted
            chunksize (int): dask chunk size. (default: min(1e6, len(df) // num_devices))
    """
    if chunksize is None:
        chunksize = min(
            int(1e6), len(df) // cupy.cuda.runtime.getDeviceCount()
        )
    if isinstance(df, cudf.DataFrame):
        df = dask_cudf.from_cudf(df, chunksize=chunksize)
    elif isinstance(df, dask_cudf.DataFrame):
        pass
    elif isinstance(df, pd.DataFrame):
        df = cudf.from_pandas(df)
        df = dask_cudf.from_cudf(df, chunksize=chunksize)
    else:
        raise ValueError(f"DataFrameType type {type(df)} not supported")
    return df


def dynamic_import(object_path):
    """Import an object from its full path."""
    if isinstance(object_path, str):
        parent, obj_name = object_path.rsplit(".", 1)
        try:
            parent = importlib.import_module(parent)
        except ImportError:
            raise ImportError(f"Could not import {object_path}")

        return getattr(parent, obj_name)

    return object_path


def get_object_path(obj):
    return obj.__class__.__module__ + '.' + obj.__class__.__name__
