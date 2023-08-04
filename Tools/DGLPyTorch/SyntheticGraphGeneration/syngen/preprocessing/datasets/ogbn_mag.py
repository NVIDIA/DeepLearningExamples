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

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from ogb.nodeproppred import NodePropPredDataset

from syngen.configuration import SynGenDatasetFeatureSpec
from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.io_utils import dump_dataframe
from syngen.utils.types import MetaData


class OGBN_MAG_Preprocessing(BasePreprocessing):
    """
    The OGBN_MAG_Preprocessing class includes the transformation
    operation for a subset of the Microsoft Academic Graph (MAG).
    It's a heterogeneous network that contains four types of entities—papers
    (736,389 nodes), authors (1,134,649 nodes), institutions (8,740 nodes),
    and fields of study (59,965 nodes)—as well as four types of directed relations
    connecting two types of entities—an author is “affiliated with” an institution,
    an author “writes” a paper, a paper “cites” a paper, and a paper “has a topic
    of” a field of study. For more information, please check
    https://ogb.stanford.edu/docs/nodeprop/

    """

    def __init__(
            self,
            source_path: str,
            destination_path: Optional[str] = None,
            download: bool = False,
            **kwargs,
    ):
        super().__init__(source_path, destination_path, download, **kwargs)

    def download(self):
        NodePropPredDataset(name="ogbn-mag", root=self.source_path)

    def _check_files(self) -> bool:
        return True

    def transform(self, gpu=False, use_cache=False):

        tabular_operator = cudf if gpu else pd
        operator = cp if gpu else np

        if use_cache and os.path.exists(self.destination_path):
            return SynGenDatasetFeatureSpec.instantiate_from_preprocessed(self.destination_path)

        shutil.rmtree(self.destination_path, ignore_errors=True)
        os.makedirs(self.destination_path)

        dataset = NodePropPredDataset(name="ogbn-mag", root=self.source_path)[0]
        data = dataset[0]
        labels = dataset[1]["paper"]

        graph_metadata = {
            MetaData.NODES: [],
            MetaData.EDGES: [],
        }

        connections = {}

        for e, edges in data["edge_index_dict"].items():

            structural_data = pd.DataFrame(edges.T, columns=[MetaData.SRC, MetaData.DST])

            connections[e[1]] = tabular_operator.DataFrame({
                "src_id": edges[0, :],
                "dst_id": edges[1, :],
            })

            edata = data["edge_reltype"][e]

            edge_type = {
                MetaData.NAME: e[1],
                MetaData.COUNT: len(structural_data),
                MetaData.SRC_NODE_TYPE: e[0],
                MetaData.DST_NODE_TYPE: e[2],
                MetaData.DIRECTED: False,
                MetaData.FEATURES: [{
                    MetaData.NAME: 'feat',
                    MetaData.DTYPE: str(edata.dtype),
                    MetaData.FEATURE_TYPE: MetaData.CATEGORICAL,
                }],
                MetaData.FEATURES_PATH: f"{e[1]}_features.parquet",
                MetaData.STRUCTURE_PATH: f"{e[1]}_list.parquet",
            }

            dump_dataframe(tabular_operator.DataFrame(edata, columns=['feat']),
                           os.path.join(self.destination_path, edge_type[MetaData.FEATURES_PATH]))
            dump_dataframe(structural_data,
                           os.path.join(self.destination_path, edge_type[MetaData.STRUCTURE_PATH]))
            graph_metadata[MetaData.EDGES].append(edge_type)

        # paper node type
        continuous_column_names = ["feat_" + str(i) for i in range(data["node_feat_dict"]["paper"].shape[1])]
        paper_features_dataframe = tabular_operator.DataFrame(
            data["node_feat_dict"]["paper"],
            columns=continuous_column_names,
        ).astype("float32")

        paper_features_dataframe["year"] = tabular_operator.DataFrame(data["node_year"]["paper"]).astype("int32")
        paper_features_dataframe["venue"] = tabular_operator.DataFrame(labels).astype("int32")

        paper_node_type = {
            MetaData.NAME: "paper",
            MetaData.COUNT: data["num_nodes_dict"]['paper'],
            MetaData.FEATURES: [
                {
                    MetaData.NAME: name,
                    MetaData.DTYPE: str(dtype),
                    MetaData.FEATURE_TYPE:
                        MetaData.CATEGORICAL if str(dtype).startswith('int') else MetaData.CONTINUOUS,
                } for name, dtype in paper_features_dataframe.dtypes.items()
            ],
            MetaData.FEATURES_PATH: "paper.parquet",
        }
        dump_dataframe(paper_features_dataframe,
                       os.path.join(self.destination_path, paper_node_type[MetaData.FEATURES_PATH]))
        graph_metadata[MetaData.NODES].append(paper_node_type)

        # author node type
        paper_features_dataframe["paper_id"] = operator.arange(paper_features_dataframe.shape[0])

        author_feat = connections["writes"].merge(
            paper_features_dataframe,
            left_on="dst_id",
            right_on="paper_id",
            how="left"
        ).groupby("src_id", sort=True).mean()
        author_features_dataframe = author_feat[continuous_column_names]

        author_node_type = {
            MetaData.NAME: "author",
            MetaData.COUNT: data["num_nodes_dict"]['author'],
            MetaData.FEATURES: [
                {
                    MetaData.NAME: name,
                    MetaData.DTYPE: str(dtype),
                    MetaData.FEATURE_TYPE: MetaData.CONTINUOUS,
                } for name, dtype in author_features_dataframe.dtypes.items()
            ],
            MetaData.FEATURES_PATH: "author.parquet",
        }
        dump_dataframe(author_features_dataframe,
                       os.path.join(self.destination_path, author_node_type[MetaData.FEATURES_PATH]))
        graph_metadata[MetaData.NODES].append(author_node_type)

        # institution node type
        author_features_dataframe["author_id"] = operator.arange(author_features_dataframe.shape[0])
        institution_feat = connections["affiliated_with"].merge(
            author_features_dataframe,
            left_on="src_id",
            right_on="author_id"
        ).groupby("dst_id", sort=True).mean()
        institution_dataframe = institution_feat[continuous_column_names]

        institution_node_type = {
            MetaData.NAME: "institution",
            MetaData.COUNT: data["num_nodes_dict"]['institution'],
            MetaData.FEATURES: [
                {
                    MetaData.NAME: name,
                    MetaData.DTYPE: str(dtype),
                    MetaData.FEATURE_TYPE: MetaData.CONTINUOUS,
                } for name, dtype in institution_dataframe.dtypes.items()
            ],
            MetaData.FEATURES_PATH: "institution.parquet",
        }
        dump_dataframe(institution_dataframe,
                       os.path.join(self.destination_path, institution_node_type[MetaData.FEATURES_PATH]))
        graph_metadata[MetaData.NODES].append(institution_node_type)

        # field_of_study node type
        field_of_study_feat = connections["has_topic"].merge(
            paper_features_dataframe,
            left_on="src_id",
            right_on="paper_id"
        ).groupby("dst_id", sort=True).mean()
        field_of_study_dataframe = field_of_study_feat[continuous_column_names]

        field_of_study_node_type = {
            MetaData.NAME: "field_of_study",
            MetaData.COUNT: data["num_nodes_dict"]['field_of_study'],
            MetaData.FEATURES: [
                {
                    MetaData.NAME: name,
                    MetaData.DTYPE: str(dtype),
                    MetaData.FEATURE_TYPE: MetaData.CONTINUOUS,
                } for name, dtype in field_of_study_dataframe.dtypes.items()
            ],
            MetaData.FEATURES_PATH: "field_of_study.parquet",
        }
        dump_dataframe(field_of_study_dataframe,
                       os.path.join(self.destination_path, field_of_study_node_type[MetaData.FEATURES_PATH]))
        graph_metadata[MetaData.NODES].append(field_of_study_node_type)

        with open(os.path.join(self.destination_path, 'graph_metadata.json'), 'w') as f:
            json.dump(graph_metadata, f, indent=4)

        graph_metadata[MetaData.PATH] = self.destination_path
        return SynGenDatasetFeatureSpec(graph_metadata)
