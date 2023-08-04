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

import copy
import json
import os
import warnings
from typing import Dict, Optional, Union

from syngen.configuration.utils import optional_comparison, one_field_from_list_of_dicts
from syngen.utils.io_utils import load_dataframe, load_graph
from syngen.utils.types import MetaData, DataSourceInputType


class SynGenDatasetFeatureSpec(dict):
    """ SynGenDatasetFeatureSpec is an util class to simply the work with SynGen Dataset Format
        Args:
            graph_metadata (Dict): dict in SynGen Format
    """

    def __init__(self, graph_metadata: Dict):
        super().__init__(graph_metadata)

    @staticmethod
    def instantiate_from_preprocessed(path: str):
        """ Creates a SynGenDatasetFeatureSpec and checks all specified files
            Args:
                path: path to the directory with a dataset in SynGen Format
        """
        if os.path.isfile(path):
            file_path = path
            dir_path = os.path.dirname(file_path)
        elif os.path.isdir(path):
            file_path = os.path.join(path, 'graph_metadata.json')
            dir_path = path
        else:
            raise ValueError(f"expected path to existing file or directory. got {path}")

        with open(file_path, 'r') as f:
            graph_metadata = json.load(f)

        graph_metadata[MetaData.PATH] = dir_path
        config = SynGenDatasetFeatureSpec(graph_metadata)
        config.validate()
        return config

    def get_tabular_data(self, part, name, cache=False, absolute_path=None, return_cat_feats=False):
        part_info = self.get_info(part, name)

        if MetaData.FEATURES_DATA in part_info:
            return part_info[MetaData.FEATURES_DATA]

        part_features_info = part_info[MetaData.FEATURES]
        part_features_path = part_info[MetaData.FEATURES_PATH]

        if part_features_path is None:
            raise ValueError()

        if MetaData.PATH not in self:
            if absolute_path is None:
                raise ValueError("Please specify the absolute path for the feature spec: "
                                 "by passing absolute_path argument or specifying MetaData.PATH in the Feature Spec")
            else:
                self[MetaData.PATH] = absolute_path

        features_df = load_dataframe(os.path.join(self[MetaData.PATH], part_features_path),
                                     feature_info=part_features_info)
        if cache:
            part_info[MetaData.FEATURES_DATA] = features_df

        if return_cat_feats:
            cat_features = [
                feature_info[MetaData.NAME]
                for feature_info in part_info[MetaData.FEATURES]
                if feature_info[MetaData.FEATURE_TYPE] == MetaData.CATEGORICAL
            ]
            return features_df, cat_features
        return features_df

    def get_structural_data(self, edge_name, cache=False, absolute_path=None, ):
        edge_info = self.get_edge_info(edge_name)

        if MetaData.STRUCTURE_DATA in edge_info:
            return edge_info[MetaData.STRUCTURE_DATA]

        structure_path = edge_info[MetaData.STRUCTURE_PATH]

        if structure_path is None:
            raise ValueError()

        if MetaData.PATH not in self:
            if absolute_path is None:
                raise ValueError("Please specify the absolute path for the feature spec: "
                                 "by passing absolute_path argument or specifying MetaData.PATH in the Feature Spec")
            else:
                self[MetaData.PATH] = absolute_path

        graph = load_graph(os.path.join(self[MetaData.PATH], structure_path))

        if cache:
            edge_info[MetaData.STRUCTURE_DATA] = graph
        return graph

    def get_edge_info(self, name: Union[str, list], src_node_type: Optional[str] = None,
                      dst_node_type: Optional[str] = None):
        if isinstance(name, list):
            src_node_type, name, dst_node_type = name
        for edge_type in self[MetaData.EDGES]:
            if edge_type[MetaData.NAME] == name \
                    and optional_comparison(src_node_type, edge_type[MetaData.SRC_NODE_TYPE]) \
                    and optional_comparison(dst_node_type, edge_type[MetaData.DST_NODE_TYPE]):
                return edge_type

    def get_node_info(self, name: str):
        for node_type in self[MetaData.NODES]:
            if node_type[MetaData.NAME] == name:
                return node_type

    def get_info(self, part, name):
        if part == MetaData.NODES:
            return self.get_node_info(name)
        elif part == MetaData.EDGES:
            return self.get_edge_info(name)
        else:
            raise ValueError(f"unsupported FeatureSpec part expected [{MetaData.NODES}, {MetaData.EDGES}], got {part}")

    def validate(self):

        for part in [MetaData.NODES, MetaData.EDGES]:
            for part_info in self[part]:
                if part_info[MetaData.FEATURES_PATH]:
                    tab_path = os.path.join(self[MetaData.PATH], part_info[MetaData.FEATURES_PATH])
                    assert os.path.exists(tab_path), f"{part}-{part_info[MetaData.NAME]}: {tab_path} does not exist"
                    assert len(part_info[MetaData.FEATURES]) > 0, \
                        f"{part}-{part_info[MetaData.NAME]}: tabular features are not specified"

                    feature_files = one_field_from_list_of_dicts(
                        part_info[MetaData.FEATURES], MetaData.FEATURE_FILE, res_aggregator=set)

                    if len(feature_files) > 1:
                        assert os.path.isdir(tab_path), \
                            "different feature files are specified MetaData. FEATURES_PATH should be a directory"
                        for ff in feature_files:
                            ff_path = os.path.join(tab_path, ff)
                            assert os.path.exists(ff_path), \
                                f"{part}-{part_info[MetaData.NAME]}: {ff_path} does not exist"

                if part == MetaData.EDGES:
                    struct_path = os.path.join(self[MetaData.PATH], part_info[MetaData.STRUCTURE_PATH])
                    assert os.path.exists(struct_path), \
                        f"{part}-{part_info[MetaData.NAME]}: {struct_path} does not exist"

    def copy(self):
        res = {}

        keys_to_ignore = {MetaData.STRUCTURE_DATA, MetaData.FEATURES_DATA}

        for part in (MetaData.EDGES, MetaData.NODES):
            res[part] = [
                {
                    k: copy.deepcopy(v)
                    for k, v in part_info.items() if k not in keys_to_ignore
                }
                for part_info in self[part]
            ]

        return SynGenDatasetFeatureSpec(res)


class SynGenConfiguration(SynGenDatasetFeatureSpec):
    """ SynGen Configuration

    """
    def __init__(self, configuration: Dict):
        super().__init__(configuration)
        self._fill_missing_values()
        self.validate()

    def validate(self):
        if MetaData.ALIGNERS in self:
            for aligner_info in self[MetaData.ALIGNERS]:

                for edge_name in aligner_info[MetaData.EDGES]:
                    if not self.get_edge_info(edge_name)[MetaData.FEATURES_PATH].endswith(".parquet"):
                        raise ValueError("Alignment supports only .parquet files right now")

                for node_name in aligner_info[MetaData.NODES]:
                    if not self.get_node_info(node_name)[MetaData.FEATURES_PATH].endswith(".parquet"):
                        raise ValueError("Alignment supports only .parquet files right now")


    def _process_tabular_generators(self, graph_part_info, part):

        if MetaData.TABULAR_GENERATORS not in graph_part_info:
            return

        if graph_part_info[MetaData.FEATURES] == -1:
            assert len(graph_part_info[MetaData.TABULAR_GENERATORS]) == 1
            tab_gen_cfg = graph_part_info[MetaData.TABULAR_GENERATORS][0]
            assert tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.TYPE] == DataSourceInputType.CONFIGURATION

            cfg = SynGenConfiguration.instantiate_from_preprocessed(tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.PATH])
            data_source_part_info = cfg.get_info(part, tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.NAME])
            graph_part_info[MetaData.FEATURES] = data_source_part_info[MetaData.FEATURES]

        for tab_gen_cfg in graph_part_info[MetaData.TABULAR_GENERATORS]:
            if tab_gen_cfg[MetaData.FEATURES_LIST] == -1:
                assert len(graph_part_info[MetaData.TABULAR_GENERATORS]) == 1, \
                    "you may use mimic value (-1) only if you specify a single tabular generator"
                tab_gen_cfg[MetaData.FEATURES_LIST] = [f[MetaData.NAME] for f in graph_part_info[MetaData.FEATURES]]

            if tab_gen_cfg[MetaData.DATA_SOURCE][MetaData.TYPE] == DataSourceInputType.RANDOM:
                edge_features = [f[MetaData.NAME] for f in graph_part_info[MetaData.FEATURES]]
                for feature_name in tab_gen_cfg[MetaData.FEATURES_LIST]:
                    if feature_name not in edge_features:
                        graph_part_info[MetaData.FEATURES].append(
                            {
                                MetaData.NAME: feature_name,
                                MetaData.DTYPE: 'float32',
                                MetaData.FEATURE_TYPE: MetaData.CONTINUOUS,
                                # Now random generator supports only continuous features
                            }
                        )

    def _fill_missing_values(self):
        for part in [MetaData.NODES, MetaData.EDGES]:
            for part_info in self[part]:

                if MetaData.FEATURES not in part_info:
                    part_info[MetaData.FEATURES] = []
                    warnings.warn(
                        f"{part}-{part_info[MetaData.NAME]}: no {MetaData.FEATURES} specified, default is []")

                if MetaData.FEATURES_PATH not in part_info:
                    part_info[MetaData.FEATURES_PATH] = None
                    warnings.warn(
                        f"{part}-{part_info[MetaData.NAME]}: no {MetaData.FEATURES_PATH} specified, default is None")

                if MetaData.COUNT not in part_info:
                    part_info[MetaData.COUNT] = -1
                    warnings.warn(
                        f"{part}-{part_info[MetaData.NAME]}: no {MetaData.COUNT} specified, "
                        f"try to mimic based on generators data")

                self._process_tabular_generators(part_info, part)

                if part == MetaData.EDGES:

                    if MetaData.DIRECTED not in part_info:
                        part_info[MetaData.DIRECTED] = False

                    if part_info[MetaData.COUNT] == -1:

                        data_source_info = part_info[MetaData.STRUCTURE_GENERATOR][MetaData.DATA_SOURCE]

                        if data_source_info[MetaData.TYPE] == DataSourceInputType.CONFIGURATION:
                            cfg = SynGenConfiguration.instantiate_from_preprocessed(data_source_info[MetaData.PATH])
                            data_source_part_info = cfg.get_info(part, data_source_info[MetaData.NAME])
                        elif data_source_info[MetaData.TYPE] == DataSourceInputType.RANDOM:
                            raise ValueError('Can\'t fill the ')
                        else:
                            raise ValueError("unsupported structure generator datasource type")

                        if part_info[MetaData.COUNT] == -1:
                            part_info[MetaData.COUNT] = data_source_part_info[MetaData.COUNT]

    def copy(self):

        res = super().copy()

        if MetaData.ALIGNERS in self:
            res[MetaData.ALIGNERS] = copy.deepcopy(self[MetaData.ALIGNERS])

        return SynGenConfiguration(res)
