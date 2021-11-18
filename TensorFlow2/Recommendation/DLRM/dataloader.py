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
# author: Tomasz Grel (tgrel@nvidia.com)


from distributed_utils import get_device_mapping
import horovod.tensorflow as hvd
from split_binary_dataset import RawBinaryDataset, DummyDataset
import numpy as np


class Permuter:
    def __init__(self, wrapped, seed):
        self.wrapped = wrapped
        rng = np.random.default_rng(seed=seed)
        self.indices = rng.permutation(len(wrapped))
        print('Permuter created, first indices: ', self.indices[:10])

    def __getitem__(self, item):
        mapped_index = self.indices[item]
        return self.wrapped[mapped_index]

    def __len__(self):
        return len(self.wrapped)


def create_input_pipelines(FLAGS):
    if FLAGS.dataset_type == 'synthetic':
        dataset_metadata = DummyDataset.get_metadata(FLAGS)
    elif FLAGS.dataset_type == 'raw':
        dataset_metadata = RawBinaryDataset.get_metadata(FLAGS.dataset_path, FLAGS.num_numerical_features)

    multi_gpu_metadata = get_device_mapping(embedding_sizes=dataset_metadata.categorical_cardinalities,
                                            num_gpus=hvd.size(),
                                            data_parallel_bottom_mlp=FLAGS.data_parallel_bottom_mlp,
                                            experimental_columnwise_split=FLAGS.experimental_columnwise_split,
                                            num_numerical_features=FLAGS.num_numerical_features)

    local_tables = multi_gpu_metadata.rank_to_categorical_ids[hvd.rank()]
    local_table_sizes = [dataset_metadata.categorical_cardinalities[i] for i in local_tables]

    numerical_features = dataset_metadata.num_numerical_features if hvd.rank() in multi_gpu_metadata.bottom_mlp_ranks else 0

    if FLAGS.dataset_type == 'synthetic':
        train_dataset = DummyDataset(batch_size=FLAGS.batch_size,
                                     num_numerical_features=numerical_features,
                                     num_categorical_features=len(local_table_sizes),
                                     num_batches=FLAGS.synthetic_dataset_train_batches)

        test_dataset = DummyDataset(batch_size=FLAGS.valid_batch_size,
                                     num_numerical_features=numerical_features,
                                     num_categorical_features=len(local_table_sizes),
                                     num_batches=FLAGS.synthetic_dataset_valid_batches)

    elif FLAGS.dataset_type == 'raw':
        train_dataset = RawBinaryDataset(data_path=FLAGS.dataset_path,
                                         valid=False,
                                         batch_size=FLAGS.batch_size,
                                         numerical_features=numerical_features,
                                         categorical_features=local_tables,
                                         categorical_feature_sizes=dataset_metadata.categorical_cardinalities,
                                         prefetch_depth=FLAGS.prefetch_batches,
                                         drop_last_batch=True)

        if FLAGS.batch_shuffle:
            train_dataset = Permuter(train_dataset, seed=FLAGS.shuffle_seed)

        test_dataset = RawBinaryDataset(data_path=FLAGS.dataset_path,
                                        valid=True,
                                        batch_size=FLAGS.valid_batch_size,
                                        numerical_features=numerical_features,
                                        categorical_features=local_tables,
                                        categorical_feature_sizes=dataset_metadata.categorical_cardinalities,
                                        prefetch_depth=FLAGS.prefetch_batches,
                                        drop_last_batch=True)

    else:
        raise ValueError(f'Unsupported dataset type: {FLAGS.dataset_type}')

    return  train_dataset, test_dataset, dataset_metadata, multi_gpu_metadata
