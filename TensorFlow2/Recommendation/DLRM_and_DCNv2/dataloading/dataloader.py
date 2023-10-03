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

from .defaults import TRAIN_MAPPING, TEST_MAPPING
from .feature_spec import FeatureSpec
from .raw_binary_dataset import TfRawBinaryDataset, DatasetMetadata
from .synthetic_dataset import SyntheticDataset

from .split_tfrecords_multihot_dataset import SplitTFRecordsDataset


def get_dataset_metadata(dataset_path, feature_spec):
    fspec_path = os.path.join(dataset_path, feature_spec)
    feature_spec = FeatureSpec.from_yaml(fspec_path)
    dataset_metadata = DatasetMetadata(num_numerical_features=feature_spec.get_number_of_numerical_features(),
                                       categorical_cardinalities=feature_spec.get_categorical_sizes())
    return dataset_metadata


def _create_pipelines_synthetic_fspec(**kwargs):
    fspec_path = os.path.join(kwargs['dataset_path'], kwargs['feature_spec'])
    feature_spec = FeatureSpec.from_yaml(fspec_path)
    dataset_metadata = DatasetMetadata(num_numerical_features=feature_spec.get_number_of_numerical_features(),
                                       categorical_cardinalities=feature_spec.get_categorical_sizes())
    local_table_sizes = [dataset_metadata.categorical_cardinalities[i] for i in kwargs['table_ids']]

    names = feature_spec.get_categorical_feature_names()

    local_names = [names[i] for i in kwargs['table_ids']]
    local_table_hotness = [feature_spec.feature_spec[name]["hotness"] for name in local_names]
    local_table_alpha = [feature_spec.feature_spec[name]["alpha"] for name in local_names]

    print('local table sizes: ', local_table_sizes)
    print('Local table hotness: ', local_table_hotness)

    train_dataset = SyntheticDataset(batch_size=kwargs['train_batch_size'],
                                     num_numerical_features=dataset_metadata.num_numerical_features,
                                     categorical_feature_cardinalities=local_table_sizes,
                                     categorical_feature_hotness=local_table_hotness,
                                     categorical_feature_alpha=local_table_alpha,
                                     num_batches=kwargs.get('synthetic_dataset_train_batches', int(1e9)),
                                     num_workers=kwargs['world_size'],
                                     variable_hotness=False)

    test_dataset = SyntheticDataset(batch_size=kwargs['test_batch_size'],
                                    num_numerical_features=dataset_metadata.num_numerical_features,
                                    categorical_feature_cardinalities=local_table_sizes,
                                    categorical_feature_hotness=local_table_hotness,
                                    categorical_feature_alpha=local_table_alpha,
                                    num_batches=kwargs.get('synthetic_dataset_valid_batches', int(1e9)),
                                    num_workers=kwargs['world_size'],
                                    variable_hotness=False)
    return train_dataset, test_dataset


def _create_pipelines_tf_raw(**kwargs):
    fspec_path = os.path.join(kwargs['dataset_path'], kwargs['feature_spec'])
    feature_spec = FeatureSpec.from_yaml(fspec_path)

    local_categorical_names = feature_spec.cat_positions_to_names(kwargs['table_ids'])
    train_dataset = TfRawBinaryDataset(feature_spec=feature_spec,
                                       instance=TRAIN_MAPPING,
                                       batch_size=kwargs['train_batch_size'],
                                       numerical_features_enabled=True,
                                       local_categorical_feature_names=local_categorical_names,
                                       rank=kwargs['rank'],
                                       world_size=kwargs['world_size'],
                                       concat_features=kwargs['concat_features'],
                                       data_parallel_categoricals=kwargs['data_parallel_input'])

    test_dataset = TfRawBinaryDataset(feature_spec=feature_spec,
                                      instance=TEST_MAPPING,
                                      batch_size=kwargs['test_batch_size'],
                                      numerical_features_enabled=True,
                                      local_categorical_feature_names=local_categorical_names,
                                      rank=kwargs['rank'],
                                      world_size=kwargs['world_size'],
                                      concat_features=kwargs['concat_features'],
                                      data_parallel_categoricals=kwargs['data_parallel_input'])
    return train_dataset, test_dataset


def _create_pipelines_split_tfrecords(**kwargs):
    fspec_path = os.path.join(kwargs['dataset_path'], kwargs['feature_spec'])
    feature_spec = FeatureSpec.from_yaml(fspec_path)

    train_dataset = SplitTFRecordsDataset(dataset_dir=feature_spec.base_directory + '/train/',
                                          feature_ids=kwargs['table_ids'],
                                          num_numerical=feature_spec.get_number_of_numerical_features(),
                                          rank=kwargs['rank'], world_size=kwargs['world_size'],
                                          batch_size=kwargs['train_batch_size'])

    test_dataset = SplitTFRecordsDataset(dataset_dir=feature_spec.base_directory + '/test/',
                                         feature_ids=kwargs['table_ids'],
                                         num_numerical=feature_spec.get_number_of_numerical_features(),
                                         rank=kwargs['rank'], world_size=kwargs['world_size'],
                                         batch_size=kwargs['test_batch_size'])

    return train_dataset, test_dataset


def create_input_pipelines(dataset_type, dataset_path, train_batch_size, test_batch_size,
                           table_ids, feature_spec, rank=0, world_size=1, concat_features=False,
                           data_parallel_input=False):

    # pass along all arguments except dataset type
    kwargs = locals()
    del kwargs['dataset_type']

    #hardcoded for now
    kwargs['synthetic_dataset_use_feature_spec'] = True
    if dataset_type == 'synthetic' and not kwargs['synthetic_dataset_use_feature_spec']:
        return _create_pipelines_synthetic(**kwargs)
    elif dataset_type == 'synthetic' and kwargs['synthetic_dataset_use_feature_spec']:  # synthetic based on feature spec
        return _create_pipelines_synthetic_fspec(**kwargs)
    elif dataset_type == 'tf_raw':
        return _create_pipelines_tf_raw(**kwargs)
    elif dataset_type == 'split_tfrecords':
        return _create_pipelines_split_tfrecords(**kwargs)
    else:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
