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
from typing import List, Dict


class FeatureSpec:
    def __init__(self, feature_spec, source_spec, channel_spec, metadata, base_directory):
        self.feature_spec: Dict = feature_spec
        self.source_spec: Dict = source_spec
        self.channel_spec: Dict = channel_spec
        self.metadata: Dict = metadata
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
