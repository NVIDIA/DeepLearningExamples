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

import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import json

from distributed_embeddings.python.layers import dist_model_parallel as dmp

from utils.checkpointing import get_variable_path

from .embedding import EmbeddingInitializer, DualEmbeddingGroup


sparse_model_parameters = ['use_mde_embeddings', 'embedding_dim', 'column_slice_threshold',
                           'embedding_zeros_initializer', 'embedding_trainable', 'categorical_cardinalities',
                           'concat_embedding', 'cpu_offloading_threshold_gb',
                           'data_parallel_input', 'row_slice_threshold', 'data_parallel_threshold']

def _gigabytes_to_elements(gb, dtype=tf.float32):
    if gb is None:
        return None

    if dtype == tf.float32:
        bytes_per_element = 4
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')

    return gb * 10**9 / bytes_per_element

class SparseModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SparseModel, self).__init__()

        sparse_model_kwargs = {k:kwargs[k] for k in sparse_model_parameters}
        for field in sparse_model_kwargs.keys():
            self.__dict__[field] = kwargs[field]

        self.num_all_categorical_features = len(self.categorical_cardinalities)
        self.use_concat_embedding = self.concat_embedding and (hvd.size() == 1) and \
                                     all(dim == self.embedding_dim[0] for dim in self.embedding_dim)
        self._create_embeddings()

    def _create_embeddings(self):
        self.embedding_layers = []

        initializer_cls = tf.keras.initializers.Zeros if self.embedding_zeros_initializer else EmbeddingInitializer

        # use a concatenated embedding for singleGPU when all embedding dimensions are equal
        if self.use_concat_embedding:
            self.embedding = DualEmbeddingGroup(cardinalities=self.categorical_cardinalities,
                                                output_dim=self.embedding_dim[0],
                                                memory_threshold=self.cpu_offloading_threshold_gb,
                                                trainable=self.trainable,
                                                use_mde_embeddings=self.use_mde_embeddings)
            return

        for table_size, dim in zip(self.categorical_cardinalities, self.embedding_dim):
            if hvd.rank() == 0:
                print(f'Creating embedding with size: {table_size} {dim}')
            e = tf.keras.layers.Embedding(input_dim=table_size, output_dim=dim,
                                          embeddings_initializer=initializer_cls())
            self.embedding_layers.append(e)

        gpu_size = _gigabytes_to_elements(self.cpu_offloading_threshold_gb)
        self.embedding = dmp.DistributedEmbedding(self.embedding_layers,
                                                  strategy='memory_balanced',
                                                  dp_input=self.data_parallel_input,
                                                  column_slice_threshold=self.column_slice_threshold,
                                                  row_slice_threshold=self.row_slice_threshold,
                                                  data_parallel_threshold=self.data_parallel_threshold,
                                                  gpu_embedding_size=gpu_size)

    def get_local_table_ids(self, rank):
        if self.use_concat_embedding or self.data_parallel_input:
            return list(range(self.num_all_categorical_features))
        else:
            return self.embedding.strategy.input_ids_list[rank]

    @tf.function
    def call(self, cat_features):
        embedding_outputs = self._call_embeddings(cat_features)
        return embedding_outputs

    def _call_embeddings(self, cat_features):
        if self.use_concat_embedding:
            x = self.embedding(cat_features)
        else:
            x = self.embedding(cat_features)
            x = tf.concat(x, axis=1)

        x = tf.cast(x, dtype=self.compute_dtype)
        return x

    def force_initialization(self, global_batch_size=64):
        categorical_features = [tf.zeros(shape=[global_batch_size, 1], dtype=tf.int32)
                                for _ in range(len(self.get_local_table_ids(hvd.rank())))]
        _ = self(categorical_features)

    def save_checkpoint(self, checkpoint_path):
        print('Gathering the embedding weights...')
        full_embedding_weights = self.embedding.get_weights()
        print('Saving the embedding weights...')
        for i, weight in enumerate(full_embedding_weights):
            filename = get_variable_path(checkpoint_path, f'feature_{i}')
            np.save(file=filename, arr=weight)
        print('Embedding checkpoint saved.')

    def load_checkpoint(self, checkpoint_path):
        self.force_initialization()
        paths = []
        for i in range(self.num_all_categorical_features):
            path = get_variable_path(checkpoint_path, f'feature_{i}')
            paths.append(path)

        self.embedding.set_weights(weights=paths)

    def save_config(self, path):
        config = {k : self.__dict__[k] for k in sparse_model_parameters}
        with open(path, 'w') as f:
            json.dump(obj=config, fp=f, indent=4)

    @staticmethod
    def from_config(path):
        with open(path) as f:
            config = json.load(fp=f)
        if 'data_parallel_input' not in config:
            config['data_parallel_input'] = False
        if 'row_slice_threshold' not in config:
            config['row_slice_threshold'] = None
        if 'data_parallel_threshold' not in config:
            config['data_parallel_threshold'] = None
        return SparseModel(**config)
