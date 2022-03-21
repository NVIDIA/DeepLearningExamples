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
from defaults import TRAIN_MAPPING, TEST_MAPPING
from distributed_utils import get_device_mapping
import horovod.tensorflow as hvd
from feature_spec import FeatureSpec
from datasets import TfRawBinaryDataset, DummyDataset, DatasetMetadata


def create_input_pipelines(flags):
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

    if flags.columnwise_split and not flags.data_parallel_bottom_mlp and dataset_metadata.num_numerical_features > 0:
        raise ValueError('Currently when using the --columnwise_split option '
                         'you must either set --data_parallel_bottom_mlp or have no numerical features')

    multi_gpu_metadata = get_device_mapping(embedding_sizes=dataset_metadata.categorical_cardinalities,
                                            num_gpus=hvd.size(),
                                            data_parallel_bottom_mlp=flags.data_parallel_bottom_mlp,
                                            columnwise_split=flags.columnwise_split,
                                            num_numerical_features=dataset_metadata.num_numerical_features)

    local_tables = multi_gpu_metadata.rank_to_categorical_ids[hvd.rank()]

    local_numerical_features_enabled = hvd.rank() in multi_gpu_metadata.bottom_mlp_ranks
    local_numerical_features = dataset_metadata.num_numerical_features if local_numerical_features_enabled else 0

    if flags.dataset_type == 'synthetic':
        local_table_sizes = [dataset_metadata.categorical_cardinalities[i] for i in local_tables]
        train_dataset = DummyDataset(batch_size=flags.batch_size,
                                     num_numerical_features=local_numerical_features,
                                     categorical_feature_cardinalities=local_table_sizes,
                                     num_batches=flags.synthetic_dataset_train_batches)

        test_dataset = DummyDataset(batch_size=flags.valid_batch_size,
                                    num_numerical_features=local_numerical_features,
                                    categorical_feature_cardinalities=local_table_sizes,
                                    num_batches=flags.synthetic_dataset_valid_batches)

    elif flags.dataset_type == 'tf_raw':
        local_categorical_feature_names = feature_spec.cat_positions_to_names(local_tables)
        train_dataset = TfRawBinaryDataset(feature_spec=feature_spec,
                                           instance=TRAIN_MAPPING,
                                           batch_size=flags.batch_size,
                                           numerical_features_enabled=local_numerical_features_enabled,
                                           local_categorical_feature_names=local_categorical_feature_names)

        test_dataset = TfRawBinaryDataset(feature_spec=feature_spec,
                                          instance=TEST_MAPPING,
                                          batch_size=flags.valid_batch_size,
                                          numerical_features_enabled=local_numerical_features_enabled,
                                          local_categorical_feature_names=local_categorical_feature_names)

    else:
        raise ValueError(f'Unsupported dataset type: {flags.dataset_type}')

    return train_dataset, test_dataset, dataset_metadata, multi_gpu_metadata
