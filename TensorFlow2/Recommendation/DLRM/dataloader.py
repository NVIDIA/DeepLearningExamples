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
#
# author: Tomasz Grel (tgrel@nvidia.com), Tomasz Cheda (tcheda@nvidia.com)

import os
import horovod.tensorflow as hvd

from defaults import TRAIN_MAPPING, TEST_MAPPING
from feature_spec import FeatureSpec
from datasets import TfRawBinaryDataset, DummyDataset, DatasetMetadata


def get_dataset_metadata(flags):
    if flags.dataset_type == 'synthetic' and not flags.synthetic_dataset_use_feature_spec:
        cardinalities = [int(d) for d in flags.synthetic_dataset_cardinalities]
        feature_spec = FeatureSpec.get_default_feature_spec(
            number_of_numerical_features=flags.synthetic_dataset_num_numerical_features,
            categorical_feature_cardinalities=cardinalities)
    else:  # synthetic based on feature spec, or raw
        fspec_path = os.path.join(flags.dataset_path, flags.feature_spec)
        feature_spec = FeatureSpec.from_yaml(fspec_path)

    dataset_metadata = DatasetMetadata(num_numerical_features=feature_spec.get_number_of_numerical_features(),
                                       categorical_cardinalities=feature_spec.get_categorical_sizes())
    return dataset_metadata


def create_input_pipelines(flags, table_ids):
    if flags.dataset_type == 'synthetic' and not flags.synthetic_dataset_use_feature_spec:
        cardinalities = [int(d) for d in flags.synthetic_dataset_cardinalities]
        feature_spec = FeatureSpec.get_default_feature_spec(
            number_of_numerical_features=flags.synthetic_dataset_num_numerical_features,
            categorical_feature_cardinalities=cardinalities)
    else:  # synthetic based on feature spec, or raw
        fspec_path = os.path.join(flags.dataset_path, flags.feature_spec)
        feature_spec = FeatureSpec.from_yaml(fspec_path)

    dataset_metadata = DatasetMetadata(num_numerical_features=feature_spec.get_number_of_numerical_features(),
                                       categorical_cardinalities=feature_spec.get_categorical_sizes())

    if flags.dataset_type == 'synthetic':
        local_table_sizes = [dataset_metadata.categorical_cardinalities[i] for i in table_ids]
        train_dataset = DummyDataset(batch_size=flags.batch_size,
                                     num_numerical_features=dataset_metadata.num_numerical_features,
                                     categorical_feature_cardinalities=local_table_sizes,
                                     num_batches=flags.synthetic_dataset_train_batches,
                                     num_workers=hvd.size())

        test_dataset = DummyDataset(batch_size=flags.batch_size,
                                    num_numerical_features=dataset_metadata.num_numerical_features,
                                    categorical_feature_cardinalities=local_table_sizes,
                                    num_batches=flags.synthetic_dataset_valid_batches,
                                    num_workers=hvd.size())

    elif flags.dataset_type == 'tf_raw':
        local_categorical_feature_names = feature_spec.cat_positions_to_names(table_ids)
        train_dataset = TfRawBinaryDataset(feature_spec=feature_spec,
                                           instance=TRAIN_MAPPING,
                                           batch_size=flags.batch_size,
                                           numerical_features_enabled=True,
                                           local_categorical_feature_names=local_categorical_feature_names,
                                           rank=hvd.rank(),
                                           world_size=hvd.size())

        test_dataset = TfRawBinaryDataset(feature_spec=feature_spec,
                                          instance=TEST_MAPPING,
                                          batch_size=flags.batch_size,
                                          numerical_features_enabled=True,
                                          local_categorical_feature_names=local_categorical_feature_names,
                                          rank = hvd.rank(),
                                          world_size = hvd.size())

    else:
        raise ValueError(f'Unsupported dataset type: {flags.dataset_type}')

    return train_dataset, test_dataset
