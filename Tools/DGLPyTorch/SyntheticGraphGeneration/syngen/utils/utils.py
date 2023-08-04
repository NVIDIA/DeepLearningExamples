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
from pathlib import PosixPath
from typing import Optional, Union

import cudf
import cupy
import dask.dataframe as dd
import dask_cudf
import cupy as cp
import numpy as np
import pandas as pd
import os

from syngen.utils.types import DataFrameType, NDArray

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

    def __init__(self, path: Optional[Union[PosixPath, str]] = str, verbose: bool = False):
        self.path = path
        self.verbose = verbose
        self.timers = {}
        self.f = None
        if self.path:
            self.f = open(self.path, "w")

    def start_counter(self, key: str):
        self.timers[key] = time.perf_counter()

    def end_counter(self, key: str, msg: str):
        end = time.perf_counter()
        start = self.timers.get(key, None)

        if start is None:
            return
        message_string = f"{msg}: {end - start:.2f}\n"
        if self.f:
            self.f.write(message_string)
        if self.verbose:
            print(message_string, end='')

    def maybe_close(self):
        if self.f:
            self.f.close()


def current_ms_time():
    return round(time.time() * 1000)


def to_ndarray(df: DataFrameType) -> NDArray:
    """ Returns potentially distributed data frame to its in-memory equivalent array. """
    if isinstance(df, (cudf.DataFrame, pd.DataFrame)):
        return df.values
    elif isinstance(df, (dask_cudf.DataFrame, dd.DataFrame)):
        return df.compute().values
    else:
        raise NotImplementedError(f'Conversion of type {type(df)} is not supported')


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
                    chunksize: Optional[int] = None):
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


def ensure_path(path: Union[str, PosixPath]):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path

def infer_operator(ndarray: NDArray):
    """ Returns array backend module (numpy or cupy). """
    if isinstance(ndarray, np.ndarray):
        return np
    elif isinstance(ndarray, cp.ndarray):
        return cp
    else:
        logger.warning(
            'Detected array of type %s, while one of (%s) was expected. Defaulting to using numpy',
            type(ndarray), 'numpy.ndarray, cupy.ndarray',
        )
        return np
