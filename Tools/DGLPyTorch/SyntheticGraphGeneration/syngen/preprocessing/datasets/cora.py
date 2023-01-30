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
import random
import subprocess
from pathlib import Path, PosixPath
from typing import List, Union

import cudf
import dask_cudf
import pandas as pd

from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


class CORAPreprocessing(BasePreprocessing):
    def __init__(
        self,
        cached: bool = False,
        nrows: int = None,
        drop_cols: list = [],
        **kwargs,
    ):
        """
        preprocessing for https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

        Args:
            user_ids (List[int]): list of users to filter dataframe
            cached (bool): skip preprocessing and use cached files
            data_path (str): path to file containing data
            nrows (int): number of rows to load from dataframe
        """
        super().__init__(cached, nrows, drop_cols)

        self.graph_info = {
            MetaData.EDGE_DATA: {
                MetaData.SRC_NAME: "src",
                MetaData.SRC_COLUMNS: ["src"],
                MetaData.DST_NAME: "dst",
                MetaData.DST_COLUMNS: ["dst"],
                MetaData.CONTINUOUS_COLUMNS: [],
                MetaData.CATEGORICAL_COLUMNS: ["src", "dst"],
            },
            MetaData.NODE_DATA: {
                MetaData.NODE_ID: "id",
                MetaData.CONTINUOUS_COLUMNS: [],
                MetaData.CATEGORICAL_COLUMNS: [],
            },
            "undirected": True,
        }

    def load_data(self, data_path: str, transform_name=None, **kwargs):
        """
            load the cora dataset
            files -
                cora.content
                cora.cites
            Args:
                data_path (str) : path to the directory containing cora dataset files
        """
        reader = kwargs.get("reader", pd)
        data_path = Path(data_path)

        cora_feat_df, _ = self.parse_cora_content(
            data_path / "cora.content", reader=reader
        )
        cora_graph_df = reader.read_csv(data_path / "cora.cites")
        cora_graph_df.columns = ["src", "dst"]
        return (
            {
                MetaData.EDGE_DATA: cora_graph_df,
                MetaData.NODE_DATA: cora_feat_df,
            },
            False,
        )

    def parse_cora_content(self, in_file, train_test_ratio=1.0, reader=pd):
        """
            This function parses Cora content (in TSV), converts string labels to integer
            label IDs, randomly splits the data into training and test sets, and returns
            the training and test sets as outputs.

            Args:
                in_file: A string indicating the input file path.
                train_percentage: A float indicating the percentage of training examples
                over the dataset.
            Returns:
                train_examples: A dict with keys being example IDs (string) and values being
                `dict` instances.
                test_examples: A dict with keys being example IDs (string) and values being
                `dict` instances.
        """
        random.seed(1)
        train_examples = {}
        test_examples = {}
        with open(in_file, "r") as cora_content:
            for line in cora_content:
                entries = line.rstrip("\n").split("\t")
                # entries contains [ID, Word1, Word2, ..., Label]; "Words" are 0/1 values.
                words = list(map(int, entries[1:-1]))
                example_id = int(entries[0])
                label = entries[-1]
                features = {
                    "id": example_id,
                    "label": label,
                }
                for i, w in enumerate(words):
                    features[f"w_{i}"] = w
                if (
                    random.uniform(0, 1) <= train_test_ratio
                ):  # for train/test split.
                    train_examples[example_id] = features
                else:
                    test_examples[example_id] = features

            self.graph_info[MetaData.NODE_DATA][
                MetaData.CATEGORICAL_COLUMNS
            ].extend([f"w_{i}" for i in range(len(words))])
            self.graph_info[MetaData.NODE_DATA][
                MetaData.CATEGORICAL_COLUMNS
            ].extend(["id", "label"])

        # TODO replace with reader.Dataframe after cudf 22.12 will be stable
        train = pd.DataFrame.from_dict(
            train_examples, orient="index"
        ).reset_index(drop=True)
        if reader != pd:
            train = reader.from_pandas(train)

        test = pd.DataFrame.from_dict(
            test_examples, orient="index"
        ).reset_index(drop=True)

        if reader != pd:
            test = reader.from_pandas(test)

        return train, test

    def download(self, data_path: Union[PosixPath, str]):
        log.info("downloading CORA dataset...")
        cmds = [
            fr"mkdir -p {data_path}",
            fr"wget 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz' -P {data_path}",
            fr"tar -xf {data_path}/cora.tgz -C {data_path}",
            fr"sed -i 's/\t/,/g' {data_path}/cora/cora.cites",
            fr"sed -i '1s/^/src,dst\n/' {data_path}/cora/cora.cites",
        ]
        for cmd in cmds:
            try:
                subprocess.check_output(cmd, shell=True)
            except subprocess.CalledProcessError as e:
                raise Exception(e.output)

    def inverse_transform(self, data):
        return data

    def transform_graph(self, data):
        """Preprocess dataset

        Args:
            data (`cudf.DartaFrame`): dataframe
        Returns:
            data (`cudf.DataFrame`): dataframe containing preprocessed data
        """
        categorical_columns = self.graph_info[MetaData.EDGE_DATA][
            MetaData.CATEGORICAL_COLUMNS
        ]
        continuous_columns = self.graph_info[MetaData.EDGE_DATA][
            MetaData.CONTINUOUS_COLUMNS
        ]

        continuous_columns = [
            c
            for c in data[MetaData.EDGE_DATA].columns
            if c in continuous_columns
        ]
        categorical_columns = [
            c
            for c in data[MetaData.EDGE_DATA].columns
            if c in categorical_columns
        ]

        for col in categorical_columns:
            data[MetaData.EDGE_DATA][col] = (
                data[MetaData.EDGE_DATA][col].astype("category").cat.codes
            )
            data[MetaData.EDGE_DATA][col] = data[MetaData.EDGE_DATA][
                col
            ].astype(int)
        columns_to_select = categorical_columns + continuous_columns
        data[MetaData.EDGE_DATA] = data[MetaData.EDGE_DATA][columns_to_select]

        categorical_columns = self.graph_info[MetaData.NODE_DATA][
            MetaData.CATEGORICAL_COLUMNS
        ]
        continuous_columns = self.graph_info[MetaData.NODE_DATA][
            MetaData.CONTINUOUS_COLUMNS
        ]

        continuous_columns = [
            c
            for c in data[MetaData.NODE_DATA].columns
            if c in continuous_columns
        ]
        categorical_columns = [
            c
            for c in data[MetaData.NODE_DATA].columns
            if c in categorical_columns
        ]

        for col in categorical_columns:
            data[MetaData.NODE_DATA][col] = (
                data[MetaData.NODE_DATA][col].astype("category").cat.codes
            )
            data[MetaData.NODE_DATA][col] = data[MetaData.NODE_DATA][
                col
            ].astype(int)
        columns_to_select = categorical_columns + continuous_columns
        data[MetaData.NODE_DATA] = data[MetaData.NODE_DATA][columns_to_select]
        return data
