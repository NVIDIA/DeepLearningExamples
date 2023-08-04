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

import logging
import os
from collections import defaultdict

from pathlib import Path, PosixPath
from typing import Optional, Union, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from syngen.utils.utils import infer_operator
from syngen.utils.types import NDArray
from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


def dump_dataframe(data: pd.DataFrame, save_path: Union[PosixPath, str], format: Optional[str] = 'parquet') -> None:

    if save_path.endswith('.csv'):
        format = 'csv'
    if save_path.endswith('.parquet'):
        format = 'parquet'

    log.info(f"writing to file {save_path} {format}")

    if format == 'parquet':
        data.to_parquet(save_path, compression=None, index=False)
    elif format == 'csv':
        data.to_csv(save_path, index=False)
    else:
        raise ValueError(f'unsupported file_format: {format}, expected `csv` or `parquet`')


def dump_generated_graph(path: Union[PosixPath, str], graph: NDArray, format: str = 'npy') -> None:
    operator = infer_operator(graph)

    if path.endswith('.npy'):
        format = 'npy'
    if path.endswith('.csv'):
        format = 'csv'
    if path.endswith('.parquet'):
        format = 'parquet'

    if format is None:
        raise ValueError()

    if format == 'npy':
        operator.save(path, graph)
    elif format == 'csv':
        operator.savetxt(path, graph, fmt='%i', delimiter='\t')
    elif format == 'parquet':
        dump_dataframe(pd.DataFrame(graph, columns=['src', 'dst'], copy=False), path)
    else:
        raise ValueError(f'unsupported file_format: {format}, expected `npy`, `parquet` or `csv`')


def merge_dataframe_files(file_paths: List[Union[PosixPath, str]], format='csv') -> pd.DataFrame:
    if format == 'parquet':
        dfs = [pd.read_parquet(fn) for fn in file_paths]
    elif format == 'csv':
        dfs = [pd.read_csv(fn) for fn in file_paths]
    else:
        raise ValueError(f'unsupported file_format: {format}, expected `csv` or `parquet`')
    return pd.concat(dfs, axis=0, ignore_index=True)


def load_dataframe(path: Union[PosixPath, str], format: Optional[str] = None, feature_info: Optional = None) -> pd.DataFrame:

    if path.endswith('.parquet'):
        format = 'parquet'
    elif path.endswith('.csv'):
        format = 'csv'
    elif path.endswith('.npy'):
        format = 'npy'
    elif os.path.isdir(path):
        format = 'dir'

    if format is None:
        raise ValueError()

    if format == 'parquet':
        return pd.read_parquet(path)
    if format == 'csv':
        return pd.read_csv(path)

    assert feature_info is not None, '`npy` and `dir` require specified feature_info'

    if format == 'npy':
        return pd.DataFrame(np.load(path, mmap_mode='r'), columns=[f[MetaData.NAME] for f in feature_info], copy=False)
    if format == 'dir':
        file_names_to_features = defaultdict(list)
        for fi in feature_info:
            file_names_to_features[fi[MetaData.FEATURE_FILE]].append(fi)
        return pd.concat(
            [load_dataframe(os.path.join(path, fn), feature_info=file_names_to_features[fn])
             for fn in os.listdir(path)], axis=1, copy=False)


def load_graph(path: Union[str, PosixPath], format: Optional[str] = None) -> np.ndarray:

    if path.endswith('.parquet'):
        format = 'parquet'
    elif path.endswith('.csv'):
        format = 'csv'
    elif path.endswith('.npy'):
        format = 'npy'

    if format is None:
        raise ValueError()

    if format == 'parquet':
        return pd.read_parquet(path).values
    if format == 'csv':
        return pd.read_csv(path).values
    if format == 'npy':
        return np.load(path, mmap_mode='c')


def write_csv_file_listener(save_path: Union[str, PosixPath], save_name: str, queue):
    KILL_SIG = "kill"
    save_path = Path(save_path) / f"{save_name}.csv"
    first_file = True
    while True:
        # - keep listening until `kill` signal
        m = queue.get()
        if m == KILL_SIG:
            break
        elif type(m) == pd.DataFrame:
            if first_file:
                m.to_csv(save_path, index=False, header=True)
                first_file = False
            else:
                m.to_csv(save_path, mode="append", index=False, header=False)
        else:
            raise Exception(f"{m} is not supported")


def merge_csv_files(
    file_paths: List[Union[str, PosixPath]],
    save_path: Union[str, PosixPath],
    save_name: str = "samples",
    header: bool = True,
    remove_original_files: bool = True,
) -> None:

    """
    Merges CSV files into a single large CSV file

    Args:
        file_paths (str): a list of paths to individual csv files
        save_path (str): a path to directory to save merged csv file
        save_name (str): file name of merged csv file
        Returns:
        None
    """

    save_path = Path(save_path)
    record_header = False

    if header:
        record_header = True

    with open(save_path / f"{save_name}", "w") as out_file:
        for i, fp in enumerate(tqdm(file_paths)):
            with open(fp, "r") as csv:
                for i, l in enumerate(csv):
                    if i == 0 and record_header:
                        out_file.write(l + "\n")
                        record_header = False
                        continue
                    elif i == 0:
                        continue
                    else:
                        out_file.write(l + "\n")

    if remove_original_files:
        for f in file_paths:
            os.remove(f)
