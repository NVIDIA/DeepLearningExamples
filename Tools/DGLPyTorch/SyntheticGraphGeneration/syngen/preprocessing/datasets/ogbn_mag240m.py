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
import shutil
from typing import Optional

import numpy as np
import pandas as pd

from ogb.lsc import MAG240MDataset

from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.configuration import SynGenDatasetFeatureSpec
from syngen.utils.io_utils import dump_dataframe
from syngen.utils.types import MetaData


class MAG240mPreprocessing(BasePreprocessing):

    def __init__(
            self,
            source_path: str,
            destination_path: Optional[str] = None,
            download: bool = False,
            skip_node_features=False,
            **kwargs,
    ):
        super().__init__(source_path, destination_path, download, **kwargs)
        self.include_node_features = not skip_node_features

    def download(self):
        MAG240MDataset(root=self.source_path)

    def _check_files(self) -> bool:
        return True

    def transform(self, gpu=False, use_cache=False):

        if gpu:
            raise ValueError("MAG240m support does not support gpu preprocessing at the moment")

        if use_cache and os.path.exists(self.destination_path):
            return SynGenDatasetFeatureSpec.instantiate_from_preprocessed(self.destination_path)

        shutil.rmtree(self.destination_path, ignore_errors=True)
        os.makedirs(self.destination_path)

        dataset = MAG240MDataset(root=self.source_path)

        graph_metadata = {
            MetaData.NODES: [],
            MetaData.EDGES: [],
        }

        # paper node type

        features = []
        features_path = None
        if self.include_node_features:
            features_path = 'paper_tabular_features'
            os.makedirs(os.path.join(self.destination_path, features_path))
            column_names = ["feat_" + str(i) for i in range(0, dataset.num_paper_features)]
            feat_memmap = dataset.paper_feat

            features = [
                {
                    MetaData.NAME: name,
                    MetaData.DTYPE: str(feat_memmap.dtype),
                    MetaData.FEATURE_TYPE: MetaData.CONTINUOUS,
                    MetaData.FEATURE_FILE: 'paper_feats.npy'
                } for name in column_names
            ]
            np.save(os.path.join(self.destination_path, features_path, 'paper_feats.npy'), feat_memmap)

            features.append({
                MetaData.NAME: 'year',
                MetaData.DTYPE: "int32",
                MetaData.FEATURE_TYPE: MetaData.CATEGORICAL,
                MetaData.FEATURE_FILE: 'year_label.npy'
            })
            features.append({
                MetaData.NAME: 'label',
                MetaData.DTYPE: "int32",
                MetaData.FEATURE_TYPE: MetaData.CATEGORICAL,
                MetaData.FEATURE_FILE: 'year_label.npy'
            })
            year_label_df = pd.DataFrame()
            year_label_df['year'] = dataset.all_paper_year
            year_label_df['label'] = np.nan_to_num(dataset.all_paper_label, nan=-2)
            np.save(os.path.join(self.destination_path, features_path, 'year_label.npy'), year_label_df.values)
            del year_label_df

        paper_node_type = {
            MetaData.NAME: "paper",
            MetaData.COUNT: dataset.num_papers,
            MetaData.FEATURES: features,
            MetaData.FEATURES_PATH: features_path,
        }
        graph_metadata[MetaData.NODES].append(paper_node_type)

        # author node type
        author_node_type = {
            MetaData.NAME: "author",
            MetaData.COUNT: dataset.num_authors,
            MetaData.FEATURES_PATH: None,
        }
        graph_metadata[MetaData.NODES].append(author_node_type)

        # institution node type
        institution_node_type = {
            MetaData.NAME: "institution",
            MetaData.COUNT: dataset.num_institutions,
            MetaData.FEATURES_PATH: None,
        }
        graph_metadata[MetaData.NODES].append(institution_node_type)

        for (src_node_type, dst_node_type), edge_name in dataset.__rels__.items():
            edges = dataset.edge_index(src_node_type, dst_node_type)
            structural_data = pd.DataFrame(edges.T, columns=[MetaData.SRC, MetaData.DST])

            edge_type = {
                MetaData.NAME: edge_name,
                MetaData.COUNT: len(structural_data),
                MetaData.SRC_NODE_TYPE: src_node_type,
                MetaData.DST_NODE_TYPE: dst_node_type,
                MetaData.DIRECTED: False,
                MetaData.FEATURES: [],
                MetaData.FEATURES_PATH: None,
                MetaData.STRUCTURE_PATH: f"{edge_name}_list.parquet",
            }
            dump_dataframe(structural_data,
                           os.path.join(self.destination_path, edge_type[MetaData.STRUCTURE_PATH]))
            graph_metadata[MetaData.EDGES].append(edge_type)

        with open(os.path.join(self.destination_path, 'graph_metadata.json'), 'w') as f:
            json.dump(graph_metadata, f, indent=4)

        graph_metadata[MetaData.PATH] = self.destination_path
        return SynGenDatasetFeatureSpec(graph_metadata)

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            "-snf",
            "--skip-node-features",
            action='store_true',
            help='Prepares only the structural part of the MAG240m dataset'
        )
        return parser
