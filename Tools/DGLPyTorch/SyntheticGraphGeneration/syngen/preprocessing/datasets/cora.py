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

import json
import os
import logging
import shutil
import subprocess
from typing import List, Union, Optional

import numpy as np
import pandas as pd

from syngen.configuration import SynGenDatasetFeatureSpec
from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


class CORAPreprocessing(BasePreprocessing):
    def __init__(
        self,
        source_path: str,
        destination_path: Optional[str] = None,
        download: bool = False,
        **kwargs,
    ):
        """
        preprocessing for https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
        """
        super().__init__(source_path, destination_path, download, **kwargs)

    def transform(self, gpu=False, use_cache=False):

        assert not gpu, "CORA preprocessing does not support cudf preprocessing"

        if use_cache and os.path.exists(self.destination_path):
            return SynGenDatasetFeatureSpec.instantiate_from_preprocessed(self.destination_path)
        tabular_operator = pd
        operator = np

        examples = {}

        with open(os.path.join(self.source_path, 'cora.content'), "r") as cora_content:
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
                examples[example_id] = features
        tabular_data = tabular_operator.DataFrame.from_dict(
            examples, orient="index"
        ).reset_index(drop=True)

        node_features = [
            {
                MetaData.NAME: f"w_{i}",
                MetaData.DTYPE: 'int64',
                MetaData.FEATURE_TYPE: MetaData.CATEGORICAL,
            }
            for i in range(len(words))
        ]
        node_features.extend([
            {
                MetaData.NAME: name,
                MetaData.DTYPE: 'int64',
                MetaData.FEATURE_TYPE: MetaData.CATEGORICAL,
            }
            for name in ["label"]
        ])

        for c in tabular_data.columns:
            tabular_data[c] = tabular_data[c].astype("category").cat.codes.astype(int)
        tabular_data = tabular_data.set_index('id')

        structural_data = tabular_operator.read_csv(os.path.join(self.source_path, "cora.cites"))
        structural_data.columns = ["src", "dst"]
        for c in ["src", "dst"]:
            structural_data[c] = structural_data[c].astype(int)

        paper_ids = operator.unique(operator.concatenate([
            structural_data["src"].values,
            structural_data["dst"].values,
        ]))

        mapping = operator.empty(int(paper_ids.max()) + 1, dtype=int)
        mapping[paper_ids] = operator.arange(len(paper_ids))

        for c in ["src", "dst"]:
            structural_data[c] = mapping[structural_data[c]]

        graph_metadata = {
            MetaData.NODES: [
                {
                    MetaData.NAME: "paper",
                    MetaData.COUNT: len(tabular_data),
                    MetaData.FEATURES: node_features,
                    MetaData.FEATURES_PATH: "paper.parquet",
                },
            ],
            MetaData.EDGES: [{
                MetaData.NAME: "cite",
                MetaData.COUNT: len(structural_data),
                MetaData.SRC_NODE_TYPE: "paper",
                MetaData.DST_NODE_TYPE: "paper",
                MetaData.DIRECTED: False,
                MetaData.FEATURES: [],
                MetaData.FEATURES_PATH: None,
                MetaData.STRUCTURE_PATH: "cite_edge_list.parquet",
            }]
        }
        shutil.rmtree(self.destination_path, ignore_errors=True)
        os.makedirs(self.destination_path)

        tabular_data.to_parquet(os.path.join(self.destination_path, "paper.parquet"))
        structural_data.to_parquet(os.path.join(self.destination_path, "cite_edge_list.parquet"))

        with open(os.path.join(self.destination_path, 'graph_metadata.json'), 'w') as f:
            json.dump(graph_metadata, f, indent=4)

        graph_metadata[MetaData.PATH] = self.destination_path
        return SynGenDatasetFeatureSpec(graph_metadata)

    def download(self):
        log.info("downloading CORA dataset...")
        cmds = [
            fr"mkdir -p {self.source_path}",
            fr"wget 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz' -P {self.source_path}",
            fr"tar -xf {self.source_path}/cora.tgz -C {self.source_path}",
            fr"sed -i 's/\t/,/g' {self.source_path}/cora/cora.cites",
            fr"sed -i '1s/^/src,dst\n/' {self.source_path}/cora/cora.cites",
            fr"mv {self.source_path}/cora/* {self.source_path}/.",
            fr"rm -r {self.source_path}/cora",
        ]
        for cmd in cmds:
            try:
                subprocess.check_output(cmd, shell=True)
            except subprocess.CalledProcessError as e:
                raise Exception(e.output)

    def _check_files(self):
        files = ['cora.cites', 'cora.content']
        return all(os.path.exists(os.path.join(self.source_path, file)) for file in files)
