# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import yaml
import os
from typing import Dict, List
from data.outbrain.defaults import TRAIN_MAPPING, TEST_MAPPING, ONEHOT_CHANNEL, MULTIHOT_CHANNEL, NUMERICAL_CHANNEL, LABEL_CHANNEL, MAP_FEATURE_CHANNEL, PARQUET_TYPE

TYPE_SELECTOR = "type"
FEATURES_SELECTOR = "features"
FILES_SELECTOR = "files"
DTYPE_SELECTOR = "dtype"
CARDINALITY_SELECTOR = "cardinality"
MAX_HOTNESS_SELECTOR = "max_hotness"

class FeatureSpec:
    def __init__(self, feature_spec=None, source_spec=None, channel_spec=None, metadata=None, base_directory=None):
        self.feature_spec: Dict = feature_spec if feature_spec is not None else {}
        self.source_spec: Dict = source_spec if source_spec is not None else {}
        self.channel_spec: Dict = channel_spec if channel_spec is not None else {}
        self.metadata: Dict = metadata if metadata is not None else {}
        self.base_directory: str = base_directory

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as feature_spec_file:
            base_directory = os.path.dirname(path)
            feature_spec = yaml.safe_load(feature_spec_file)
            return cls.from_dict(feature_spec, base_directory=base_directory)

    @classmethod
    def from_dict(cls, source_dict, base_directory):
        return cls(base_directory=base_directory, **source_dict)

    def to_dict(self) -> Dict:
        attributes_to_dump = ['feature_spec', 'source_spec', 'channel_spec', 'metadata']
        return {attr: self.__dict__[attr] for attr in attributes_to_dump}

    def to_string(self):
        return yaml.dump(self.to_dict())

    def to_yaml(self, output_path=None):
        if not output_path:
            output_path = self.base_directory + '/feature_spec.yaml'
        with open(output_path, 'w') as output_file:
            print(yaml.dump(self.to_dict()), file=output_file)

    def _check_one_label_feature(self):
        assert len(self.get_names_by_channel(LABEL_CHANNEL)) == 1

    def _check_all_required_channels_present(self):
        # check that channels are the ones expected
        present_channels = list(self.channel_spec.keys())
        required_channels = [ONEHOT_CHANNEL, MULTIHOT_CHANNEL, NUMERICAL_CHANNEL, LABEL_CHANNEL, MAP_FEATURE_CHANNEL]
        assert sorted(present_channels) == sorted(required_channels)

    def _check_all_used_features_are_defined(self):
        # check that all features used in channel spec are defined in feature_spec
        for channel_features in self.channel_spec.values():
            for feature in channel_features:
                assert feature in self.feature_spec

    def _check_categoricals_have_cardinality(self):
        all_categoricals = self.get_names_by_channel(ONEHOT_CHANNEL) + self.get_names_by_channel(MULTIHOT_CHANNEL)
        for feature_name in all_categoricals:
            feature_dict = self.feature_spec[feature_name]
            assert CARDINALITY_SELECTOR in feature_dict
            assert isinstance(feature_dict[CARDINALITY_SELECTOR], int)

    def _check_required_mappings_present(self):
        # check that mappings are the ones expected
        mapping_name_list = list(self.source_spec.keys())
        assert sorted(mapping_name_list) == sorted([TEST_MAPPING, TRAIN_MAPPING])

    def _check_all_chunks_are_parquet(self):
        for mapping_name in [TRAIN_MAPPING, TEST_MAPPING]:
            mapping = self.source_spec[mapping_name]
            for chunk in mapping:
                assert chunk[TYPE_SELECTOR] == PARQUET_TYPE

    def _check_only_one_chunk_per_mapping(self):
        for mapping_name in [TRAIN_MAPPING, TEST_MAPPING]:
            mapping = self.source_spec[mapping_name]
            assert len(mapping) == 1

    def _check_all_features_have_source_where_necessary(self, is_map_channel_active):
        for channel_name, channel_features in self.channel_spec.items():
            if channel_name != MAP_FEATURE_CHANNEL:
                for mapping_name in [TRAIN_MAPPING, TEST_MAPPING]:
                    # This uses the fact that we require that mappings only have one chunk here
                    features_in_mapping = set(self.source_spec[mapping_name][0][FEATURES_SELECTOR])
                    for feature in channel_features:
                        assert feature in features_in_mapping
            else:
                map_channel_features = self.get_names_by_channel(MAP_FEATURE_CHANNEL)
                if len(map_channel_features) == 1:
                    # This uses the fact that we require that mappings only have one chunk here
                    map_feature_name = map_channel_features[0]
                    test_mapping_features = set(self.source_spec[TEST_MAPPING][0][FEATURES_SELECTOR])
                    assert map_feature_name in test_mapping_features

    def _check_map_feature_selected_if_enabled(self, is_map_feature_required):
        map_channel_features = self.get_names_by_channel(MAP_FEATURE_CHANNEL)
        assert len(map_channel_features) <= 1
        if is_map_feature_required:
            assert len(map_channel_features) == 1


    def _check_dtype_correct_if_specified(self):
        # make sure that if dtype is specified, it is convertible to float32 for numerical and convertible to int64 for categorical
        # these are the requirements specified by tf.feature_column.categorical_column_with_identity and tf.feature_column.numeric_column
        categorical_features = self.get_names_by_channel(ONEHOT_CHANNEL) + self.get_names_by_channel(MULTIHOT_CHANNEL)
        categorical_allowed_types = {"int64", "int32"}
        for feature in categorical_features:
            feature_dict = self.feature_spec[feature]
            if DTYPE_SELECTOR in feature_dict:
                assert feature_dict[DTYPE_SELECTOR] in categorical_allowed_types

        numerical_features = self.get_names_by_channel(NUMERICAL_CHANNEL)
        numerical_allowed_types = {"float32", "float64"}
        for feature in numerical_features:
            feature_dict = self.feature_spec[feature]
            if DTYPE_SELECTOR in feature_dict:
                assert feature_dict[DTYPE_SELECTOR] in numerical_allowed_types

    def _check_multihots_have_hotness_specified(self):
        multihot_features = self.get_names_by_channel(MULTIHOT_CHANNEL)
        for feature_name in multihot_features:
            feature_dict = self.feature_spec[feature_name]
            assert MAX_HOTNESS_SELECTOR in feature_dict
            assert isinstance(feature_dict[MAX_HOTNESS_SELECTOR], int)

    def _check_enough_files_for_ranks(self, world_size):
        if world_size is not None:
            for mapping in self.source_spec.values():
                only_chunk = mapping[0]
                files_number = len(only_chunk[FILES_SELECTOR])
                assert files_number >= world_size, "NVTabular dataloader requires parquet to have at least as many partitions as there are workers"

    def check_feature_spec(self, require_map_channel, world_size=None):
        self._check_required_mappings_present()
        self._check_all_required_channels_present()
        self._check_one_label_feature()
        self._check_map_feature_selected_if_enabled(require_map_channel)
        self._check_all_used_features_are_defined()
        self._check_categoricals_have_cardinality()
        self._check_all_chunks_are_parquet()
        self._check_only_one_chunk_per_mapping()
        self._check_all_features_have_source_where_necessary(require_map_channel)
        self._check_dtype_correct_if_specified()
        self._check_multihots_have_hotness_specified()
        self._check_enough_files_for_ranks(world_size)

    def get_paths_by_mapping(self, mapping: str):
        paths_from_fspec = []
        chunk_list = self.source_spec[mapping]
        for chunk in chunk_list:
            paths_from_fspec.extend(chunk[FILES_SELECTOR])

        paths = [os.path.join(self.base_directory, p) for p in paths_from_fspec]
        return paths

    def get_names_by_channel(self, channel_name) -> List[str]:
        return self.channel_spec[channel_name]

    def get_multihot_hotnesses(self, multihot_features: List[str]) -> Dict[str, int]:
        return {feature_name:self.feature_spec[feature_name][MAX_HOTNESS_SELECTOR] for feature_name in multihot_features}

    def get_cardinalities(self, features: List[str]) -> Dict[str, int]:
        cardinalities = {feature_name: self.feature_spec[feature_name][CARDINALITY_SELECTOR]
                         for feature_name in features}
        return cardinalities