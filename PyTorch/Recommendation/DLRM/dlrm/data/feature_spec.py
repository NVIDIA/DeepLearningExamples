# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from typing import Dict
from typing import List
import numpy as np
from dlrm.data.defaults import CATEGORICAL_CHANNEL, NUMERICAL_CHANNEL, LABEL_CHANNEL, \
    TRAIN_MAPPING, TEST_MAPPING, \
    TYPE_SELECTOR, FEATURES_SELECTOR, FILES_SELECTOR, CARDINALITY_SELECTOR, DTYPE_SELECTOR, \
    SPLIT_BINARY, \
    get_categorical_feature_type

""" For performance reasons, numerical features are required to appear in the same order
    in both source_spec and channel_spec.
    For more detailed requirements, see the check_feature_spec method"""


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

    def get_number_of_numerical_features(self) -> int:
        numerical_features = self.channel_spec[NUMERICAL_CHANNEL]
        return len(numerical_features)

    def cat_positions_to_names(self, positions: List[int]):
        #  Ordering needs to correspond to the one in get_categorical_sizes()
        feature_names = self.get_categorical_feature_names()
        return [feature_names[i] for i in positions]

    def get_categorical_feature_names(self):
        """ Provides the categorical feature names. The returned order should me maintained."""
        return self.channel_spec[CATEGORICAL_CHANNEL]

    def get_categorical_sizes(self) -> List[int]:
        """For a given feature spec, this function is expected to return the sizes in the order corresponding to the
        order in the channel_spec section """
        categorical_features = self.get_categorical_feature_names()
        cardinalities = [self.feature_spec[feature_name][CARDINALITY_SELECTOR] for feature_name in
                         categorical_features]

        return cardinalities

    def check_feature_spec(self):
        # TODO check if cardinality fits in dtype, check if base directory is set
        # TODO split into two checking general and model specific requirements
        # check that mappings are the ones expected
        mapping_name_list = list(self.source_spec.keys())
        assert sorted(mapping_name_list) == sorted([TEST_MAPPING, TRAIN_MAPPING])

        # check that channels are the ones expected
        channel_name_list = list(self.channel_spec.keys())
        assert sorted(channel_name_list) == sorted([CATEGORICAL_CHANNEL, NUMERICAL_CHANNEL, LABEL_CHANNEL])

        categorical_features_list = self.channel_spec[CATEGORICAL_CHANNEL]
        numerical_features_list = self.channel_spec[NUMERICAL_CHANNEL]
        label_features_list = self.channel_spec[LABEL_CHANNEL]
        set_of_categorical_features = set(categorical_features_list)
        set_of_numerical_features = set(numerical_features_list)

        # check that exactly one label feature is selected
        assert len(label_features_list) == 1
        label_feature_name = label_features_list[0]

        # check that lists in channel spec contain unique names
        assert sorted(list(set_of_categorical_features)) == sorted(categorical_features_list)
        assert sorted(list(set_of_numerical_features)) == sorted(numerical_features_list)

        # check that all features used in channel spec are exactly ones defined in feature_spec
        feature_spec_features = list(self.feature_spec.keys())
        channel_spec_features = list(set.union(set_of_categorical_features,
                                               set_of_numerical_features,
                                               {label_feature_name}))
        assert sorted(feature_spec_features) == sorted(channel_spec_features)

        # check that correct dtypes are provided for all features
        for feature_dict in self.feature_spec.values():
            assert DTYPE_SELECTOR in feature_dict
            try:
                np.dtype(feature_dict[DTYPE_SELECTOR])
            except TypeError:
                assert False, "Type not understood by numpy"

        # check that categorical features have cardinality provided
        for feature_name, feature_dict in self.feature_spec.items():
            if feature_name in set_of_categorical_features:
                assert CARDINALITY_SELECTOR in feature_dict
                assert isinstance(feature_dict[CARDINALITY_SELECTOR], int)

        for mapping_name in [TRAIN_MAPPING, TEST_MAPPING]:

            mapping = self.source_spec[mapping_name]
            mapping_features = set()
            for chunk in mapping:
                # check that chunk has the correct type
                assert chunk[TYPE_SELECTOR] == SPLIT_BINARY

                contained_features = chunk[FEATURES_SELECTOR]
                containing_files = chunk[FILES_SELECTOR]

                # check that features are unique in mapping
                for feature in contained_features:
                    assert feature not in mapping_features
                    mapping_features.add(feature)

                # check that chunk has at least one features
                assert len(contained_features) >= 1

                # check that chunk has exactly file
                assert len(containing_files) == 1

                first_feature = contained_features[0]

                if first_feature in set_of_categorical_features:
                    # check that each categorical feature is in a different file
                    assert len(contained_features) == 1

                elif first_feature in set_of_numerical_features:
                    # check that numerical features are all in one chunk
                    assert sorted(contained_features) == sorted(numerical_features_list)

                    # check that ordering is exactly same as in channel spec - required for performance
                    assert contained_features == numerical_features_list

                    # check numerical dtype
                    for feature in contained_features:
                        assert np.dtype(self.feature_spec[feature][DTYPE_SELECTOR]) == np.float16

                elif first_feature == label_feature_name:
                    # check that label feature is in a separate file
                    assert len(contained_features) == 1

                    # check label dtype
                    assert np.dtype(self.feature_spec[first_feature][DTYPE_SELECTOR]) == bool

                else:
                    assert False, "Feature of unknown type"

            # check that all features appeared in mapping
            assert sorted(mapping_features) == sorted(feature_spec_features)

    @staticmethod
    def get_default_feature_spec(number_of_numerical_features, categorical_feature_cardinalities):
        numerical_feature_fstring = "num_{}"
        categorical_feature_fstring = "cat_{}.bin"
        label_feature_name = "label"

        numerical_file_name = "numerical.bin"
        categorical_file_fstring = "{}"  # TODO remove .bin from feature name, add to file name
        label_file_name = "label.bin"

        number_of_categorical_features = len(categorical_feature_cardinalities)
        numerical_feature_names = [numerical_feature_fstring.format(i) for i in range(number_of_numerical_features)]
        categorical_feature_names = [categorical_feature_fstring.format(i) for i in
                                     range(number_of_categorical_features)]
        cat_feature_types = [get_categorical_feature_type(int(cat_size)) for cat_size in
                             categorical_feature_cardinalities]

        feature_dict = {f_name: {DTYPE_SELECTOR: str(np.dtype(f_type)), CARDINALITY_SELECTOR: f_size}
                        for f_name, f_type, f_size in
                        zip(categorical_feature_names, cat_feature_types, categorical_feature_cardinalities)}
        for f_name in numerical_feature_names:
            feature_dict[f_name] = {DTYPE_SELECTOR: str(np.dtype(np.float16))}
        feature_dict[label_feature_name] = {DTYPE_SELECTOR: str(np.dtype(bool))}

        channel_spec = {CATEGORICAL_CHANNEL: categorical_feature_names,
                        NUMERICAL_CHANNEL: numerical_feature_names,
                        LABEL_CHANNEL: [label_feature_name]}
        source_spec = {}

        for filename in (TRAIN_MAPPING, TEST_MAPPING):
            source_spec[filename] = []
            dst_folder = filename

            numerical_file_path = os.path.join(dst_folder, numerical_file_name)
            source_spec[filename].append({TYPE_SELECTOR: SPLIT_BINARY,
                                          FEATURES_SELECTOR: numerical_feature_names,
                                          FILES_SELECTOR: [numerical_file_path]})

            label_file_path = os.path.join(dst_folder, label_file_name)
            source_spec[filename].append({TYPE_SELECTOR: SPLIT_BINARY,
                                          FEATURES_SELECTOR: [label_feature_name],
                                          FILES_SELECTOR: [label_file_path]})

            for feature_name in categorical_feature_names:
                categorical_file_name = categorical_file_fstring.format(feature_name)
                categorical_file_path = os.path.join(dst_folder, categorical_file_name)
                source_spec[filename].append({TYPE_SELECTOR: SPLIT_BINARY,
                                              FEATURES_SELECTOR: [feature_name],
                                              FILES_SELECTOR: [categorical_file_path]})

        return FeatureSpec(feature_spec=feature_dict, source_spec=source_spec, channel_spec=channel_spec, metadata={})

    def get_mapping_paths(self, mapping_name: str):
        label_feature_name = self.channel_spec[LABEL_CHANNEL][0]
        set_of_categorical_features = set(self.channel_spec[CATEGORICAL_CHANNEL])
        set_of_numerical_features = set(self.channel_spec[NUMERICAL_CHANNEL])

        label_path = None
        numerical_path = None
        categorical_paths = dict()
        for chunk in self.source_spec[mapping_name]:
            local_path = os.path.join(self.base_directory, chunk[FILES_SELECTOR][0])
            if chunk[FEATURES_SELECTOR][0] in set_of_numerical_features:
                numerical_path = local_path
            elif chunk[FEATURES_SELECTOR][0] in set_of_categorical_features:
                local_feature = chunk[FEATURES_SELECTOR][0]
                categorical_paths[local_feature] = local_path
            elif chunk[FEATURES_SELECTOR][0] == label_feature_name:
                label_path = local_path

        return label_path, numerical_path, categorical_paths
