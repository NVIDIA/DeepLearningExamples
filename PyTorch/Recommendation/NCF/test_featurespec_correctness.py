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

from feature_spec import FeatureSpec
from neumf_constants import TEST_SAMPLES_PER_SERIES
from dataloading import TorchTensorDataset
import torch
import os
import sys


def test_matches_template(path, template_path):
    loaded_featurespec_string = FeatureSpec.from_yaml(path).to_string()
    loaded_template_string = FeatureSpec.from_yaml(template_path).to_string()
    assert loaded_template_string == loaded_featurespec_string


def mock_args():
    class Obj:
        pass

    args = Obj()
    args.__dict__['local_rank'] = 0
    return args


def test_dtypes(path):
    loaded_featurespec = FeatureSpec.from_yaml(path)
    features = loaded_featurespec.feature_spec
    declared_dtypes = {name: data['dtype'] for name, data in features.items()}
    source_spec = loaded_featurespec.source_spec
    for mapping in source_spec.values():
        for chunk in mapping:
            chunk_dtype = None
            for present_feature in chunk['features']:
                assert present_feature in declared_dtypes, "unknown feature in mapping"
                # Check declared type
                feature_dtype = declared_dtypes[present_feature]
                if chunk_dtype is None:
                    chunk_dtype = feature_dtype
                else:
                    assert chunk_dtype == feature_dtype

            path_to_load = os.path.join(loaded_featurespec.base_directory, chunk['files'][0])
            loaded_data = torch.load(path_to_load)
            assert str(loaded_data.dtype) == chunk_dtype


def test_cardinalities(path):
    loaded_featurespec = FeatureSpec.from_yaml(path)
    features = loaded_featurespec.feature_spec
    declared_cardinalities = {name: data['cardinality'] for name, data in features.items() if 'cardinality' in data}
    source_spec = loaded_featurespec.source_spec

    for mapping_name, mapping in source_spec.items():
        dataset = TorchTensorDataset(loaded_featurespec, mapping_name, mock_args())
        for feature_name, cardinality in declared_cardinalities.items():
            feature_data = dataset.features[feature_name]
            biggest_num = feature_data.max().item()
            assert biggest_num < cardinality


def test_samples_in_test_series(path):
    loaded_featurespec = FeatureSpec.from_yaml(path)

    series_length = loaded_featurespec.metadata[TEST_SAMPLES_PER_SERIES]
    dataset = TorchTensorDataset(loaded_featurespec, 'test', mock_args())
    for feature in dataset.features.values():
        assert len(feature) % series_length == 0


if __name__ == '__main__':
    tested_spec = sys.argv[1]
    template = sys.argv[2]

    test_cardinalities(tested_spec)
    test_dtypes(tested_spec)
    test_samples_in_test_series(tested_spec)
    test_matches_template(tested_spec, template)
