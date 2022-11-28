# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict

import numpy as np
import yaml

from sim.data.defaults import (CARDINALITY_SELECTOR, DIMENSIONS_SELECTOR, DTYPE_SELECTOR, LABEL_CHANNEL,
                               NEGATIVE_HISTORY_CHANNEL, POSITIVE_HISTORY_CHANNEL, TARGET_ITEM_FEATURES_CHANNEL,
                               TEST_MAPPING, TRAIN_MAPPING, USER_FEATURES_CHANNEL)


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

    def to_dict(self):
        attributes_to_dump = ['feature_spec', 'source_spec', 'channel_spec', 'metadata']
        return {attr: self.__dict__[attr] for attr in attributes_to_dump}

    def to_string(self):
        return yaml.dump(self.to_dict())

    def to_yaml(self, output_path=None):
        if not output_path:
            output_path = self.base_directory + '/feature_spec.yaml'
        with open(output_path, 'w') as output_file:
            print(yaml.dump(self.to_dict()), file=output_file)

    @staticmethod
    def get_default_features_names(number_of_user_features, number_of_item_features):
        user_feature_fstring = 'user_feat_{}'
        item_feature_fstring = 'item_feat_{}_{}'
        label_feature_name = "label"

        item_channels_feature_name_suffixes = ['trgt', 'pos', 'neg']

        user_features_names = [user_feature_fstring.format(i) for i in range(number_of_user_features)]

        item_features_names = [item_feature_fstring.format(i, channel_suffix)
                               for channel_suffix in item_channels_feature_name_suffixes
                               for i in range(number_of_item_features)]

        return [label_feature_name] + user_features_names + item_features_names

    @staticmethod
    def get_default_feature_spec(user_features_cardinalities, item_features_cardinalities, max_seq_len):

        number_of_user_features = len(user_features_cardinalities)
        number_of_item_features = len(item_features_cardinalities)

        all_features_names = FeatureSpec.get_default_features_names(number_of_user_features, number_of_item_features)

        user_features = {
            f_name: {
                DTYPE_SELECTOR: str(np.dtype(np.int64)),
                CARDINALITY_SELECTOR: int(cardinality)
            } for i, (f_name, cardinality)
            in enumerate(zip(all_features_names[1:1+number_of_user_features], user_features_cardinalities))
        }

        item_channels = [TARGET_ITEM_FEATURES_CHANNEL, POSITIVE_HISTORY_CHANNEL, NEGATIVE_HISTORY_CHANNEL]
        item_channels_feature_dicts = [{} for _ in range(len(item_channels))]

        item_channels_info = list(zip(item_channels, item_channels_feature_dicts))

        for i, cardinality in enumerate(item_features_cardinalities):
            for j, (channel, dictionary) in enumerate(item_channels_info):

                feature_name = all_features_names[1 + number_of_user_features + i + j * number_of_item_features]

                dictionary[feature_name] = {
                    DTYPE_SELECTOR: str(np.dtype(np.int64)),
                    CARDINALITY_SELECTOR: int(cardinality)
                }

                if channel != TARGET_ITEM_FEATURES_CHANNEL:
                    dictionary[feature_name][DIMENSIONS_SELECTOR] = [max_seq_len]

        feature_spec = {
            feat_name: feat_spec
            for dictionary in [user_features] + item_channels_feature_dicts
            for feat_name, feat_spec in dictionary.items()
        }

        feature_spec[all_features_names[0]] = {DTYPE_SELECTOR: str(np.dtype(np.bool))}

        channel_spec = {
            USER_FEATURES_CHANNEL: list(user_features),
            TARGET_ITEM_FEATURES_CHANNEL: list(item_channels_feature_dicts[0]),
            POSITIVE_HISTORY_CHANNEL: list(item_channels_feature_dicts[1]),
            NEGATIVE_HISTORY_CHANNEL: list(item_channels_feature_dicts[2]),
            LABEL_CHANNEL: all_features_names[:1]
        }

        source_spec = {
            split: [
                {
                    'type': 'tfrecord',
                    'features': all_features_names,
                    'files': []
                }
            ] for split in [TRAIN_MAPPING, TEST_MAPPING]
        }

        return FeatureSpec(feature_spec=feature_spec, channel_spec=channel_spec, source_spec=source_spec)
