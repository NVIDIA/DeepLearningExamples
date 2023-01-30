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
from abc import ABC
from pathlib import Path
from typing import Dict

import cudf
import pandas as pd

from syngen.utils import write_csv
from syngen.utils.types import DataFrameType, MetaData

logger = logging.getLogger(__name__)
log = logger


class BasePreprocessing(ABC):
    """Base class for all preprocessing transforms.

       Args:
            cached (bool): skip preprocessing and use cached files
            data_path (str): path to file containing data
            nrows (int): number of rows to load from dataframe
            drop_cols (list): columns to drop from loaded data
    """

    def __init__(
        self, cached: bool = True, nrows: int = None, drop_cols: list = [],
    ):

        self.graph_info = {
            MetaData.EDGE_DATA: {
                MetaData.SRC_NAME: "src",
                MetaData.SRC_COLUMNS: ["src"],
                MetaData.DST_NAME: "dst",
                MetaData.DST_COLUMNS: ["dst"],
                MetaData.CONTINUOUS_COLUMNS: [],
                MetaData.CATEGORICAL_COLUMNS: [],
            },
            MetaData.NODE_DATA: {
                MetaData.NODE_NAME: "id",
                MetaData.NODE_COLUMNS: ["id"],
                MetaData.CONTINUOUS_COLUMNS: [],
                MetaData.CATEGORICAL_COLUMNS: [],
            },
            MetaData.UNDIRECTED: True,
        }

        self.nrows = nrows
        self.cached = cached
        self.drop_cols = []

        self.preprocessed_table = None

    def load_data(self, data_path: str, **kwargs):
        """
            loads either the raw data or preprocessed data
            assumes preprocessed data is stored in `data_dir/preprocessed`
            of original data csv file as provided in `data_path`

            Args:
                data_path (str): path to csv file
            Returns:
                data (DataFrameType): pre-processed data
                flag (bool): determining if pre-cached data was loaded
        """
        files = os.listdir(self.preprocessed_dirname)
        fname = f"{self.fname}.encoded.csv"

        files = [f for f in files if fname in f]
        reader = kwargs.get("reader", None)
        if self.cached and len(files):
            data_dict = {}
            for file in files:
                data_file = os.path.join(self.preprocessed_dirname, file)
                log.info(f"cached encoded data is read from {data_file}")
                data = self.get_csv(data_file, reader=reader)
                cols = []
                if self.drop_cols:
                    cols = [c for c in data.columns if c in self.drop_cols]
                if cols:
                    log.info(f"dropping column: {cols}")
                    data.drop(columns=cols, inplace=True)
                key = file.split(fname)[0].strip(".")
                if key == MetaData.EDGE_DATA.value:
                    data_dict[MetaData.EDGE_DATA] = data
                elif key == MetaData.NODE_DATA.value:
                    data_dict[MetaData.NODE_DATA] = data
                else:
                    raise ValueError(
                        f"Unrecgonized cached files, cannot load data"
                    )
            return data_dict, True
        data = self.get_csv(data_path, reader=reader)
        log.info(f"droping column: {self.drop_cols}")
        cols = []

        if self.drop_cols:
            cols = [c for c in data.columns if c in self.drop_cols]

        if cols:
            log.info(f"dropping column: {cols}")
            data.drop(columns=cols, inplace=True)

        return data, False

    def get_csv(self, data_path: str, reader=None):

        if reader is None:
            reader = pd
        data = reader.read_csv(data_path)
        if self.nrows is not None:
            data = data.sample(n=self.nrows).reset_index(drop=True)
        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def save_data(self, data, fname):
        log.info(
            f"writing cached csv to \
                {os.path.join(self.preprocessed_dirname, fname)}"
        )
        if not os.path.exists(self.preprocessed_dirname):
            os.mkdir(self.preprocessed_dirname)
        write_csv(data, os.path.join(self.preprocessed_dirname, fname))

    def add_graph_edge_cols(self, data: DataFrameType) -> DataFrameType:
        """Defines generic function for creating `src` and
            `dst`graph edge columns using a subset of the
            columns in the dataset.
            Assumes the `graph_info[MetaData.EDGE_DATA][MetaData.SRC_COLUMNS]` and
            `graph_info[MetaData.EDGE_DATA][MetaData.DST_COLUMNS]` provides a list of
            columns that are concatenated to create a unique node id.
        """
        edge_info = self.graph_info[MetaData.EDGE_DATA]

        # - src column
        src_name = edge_info[MetaData.SRC_NAME]
        if src_name not in data.columns:
            columns = edge_info[MetaData.SRC_COLUMNS]
            data[src_name] = data[columns[0]].astype(str)
            for i in range(1, len(columns)):
                data[src_name] += data[columns[i]].astype(str)

            cols_to_drop = set(columns) - {src_name}
            cols_to_drop = [c for c in cols_to_drop if c in data.columns]
            if cols_to_drop:
                data = data.drop(columns=cols_to_drop)

        # - dst column
        dst_name = edge_info[MetaData.DST_NAME]
        if dst_name not in data.columns:
            columns = edge_info[MetaData.DST_COLUMNS]
            data[dst_name] = data[columns[0]].astype(str)
            for i in range(1, len(columns)):
                data[dst_name] += data[columns[i]].astype(str)

            cols_to_drop = set(columns) - {dst_name}
            cols_to_drop = [c for c in cols_to_drop if c in data.columns]
            if cols_to_drop:
                data = data.drop(columns=cols_to_drop)

        return data

    def add_graph_node_cols(self, data: DataFrameType) -> DataFrameType:
        """
            defines generic function for creating graph node `id`columns using a
            subset of the columns in `data`.
            Assumes the `graph_info[MetaData.NODE_DATA][MetaData.ID_COLUMNS]` provides a list of
            categorical columns that are concatenated to create a unique node id.
        """
        node_info = self.graph_info[MetaData.NODE_DATA]
        id_col = node_info[MetaData.NODE_NAME]
        if id_col not in data.columns:
            columns = node_info[MetaData.NODE_COLUMNS]
            data[id_col] = data[columns[0]].astype(str)
            for i in range(1, len(columns)):
                data[id_col] += data[columns[i]].astype(str)

            cols_to_drop = set(columns) - {id_col}
            cols_to_drop = [c for c in cols_to_drop if c in data.columns]
            if cols_to_drop:
                data = data.drop(columns=cols_to_drop)

        return data

    def transform(
        self, data_path: str, gpu: bool = False, cast_to_pd=True
    ) -> DataFrameType:
        """
            Generic wrapper for graph transform. 
            Preprocessed data will be cached in the parent directory of `data_path`
            
            Args:
                data_path (str): data path
            
            Returns:
                Dict containing graph data
        """
        data_path = Path(data_path)
        self.data_dir = str(data_path.parent)
        self.preprocessed_dirname = os.path.join(self.data_dir, "preprocessed")
        if not os.path.exists(self.preprocessed_dirname):
            os.makedirs(self.preprocessed_dirname)
        self.fname = ".".join(str(data_path.name).split(".")[:-1])
        if gpu:
            data, preproc_flag = self.load_data(data_path, reader=cudf)
        else:
            data, preproc_flag = self.load_data(data_path)
        if preproc_flag:
            return data
        data = self.transform_graph(data)

        for k, v in data.items():
            if isinstance(v, cudf.DataFrame) and cast_to_pd:
                data[k] = v.to_pandas()
            if self.cached:
                v.to_csv(
                    os.path.join(
                        self.preprocessed_dirname,
                        f"{k}.{self.fname}.encoded.csv",
                    ),
                    index=False,
                )

        return data

    def transform_graph(self, data: DataFrameType) -> Dict[str, DataFrameType]:
        """Transform the data into graph.

        Args:
            data (DataFrameType): input data

        Returns:
            Transformed graph data.
        """
        raise NotImplementedError(
            "`transform_graph` function must be implemented"
        )

    def inverse_transform_graph(self, data):
        """Optional inverse transform to reverse preproc steps"""
        pass
