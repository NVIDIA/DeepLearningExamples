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

import math

import tensorflow as tf
import numpy as np
from distributed_embeddings.python.layers import embedding

from utils.checkpointing import get_variable_path


# write embedding checkpoints of 1M rows at a time
_embedding_checkpoint_batch = 1024 * 1024


@tf.keras.utils.register_keras_serializable()
class EmbeddingInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = tf.float32

        with tf.device('/CPU:0'):
            maxval = tf.sqrt(tf.constant(1.) / tf.cast(shape[0], tf.float32))
            maxval = tf.cast(maxval, dtype=dtype)
            minval = -maxval

            weights = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
            weights = tf.cast(weights, dtype=dtype)
        return weights

    def get_config(self):
        return {}


class Embedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, trainable=True, dtype=tf.float32, feature_name=None,
                 embeddings_initializer=None):
        super(Embedding, self).__init__(dtype=dtype)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.embedding_table = None
        self.trainable = trainable

        self.feature_name = feature_name
        if not self.feature_name:
            self.feature_name = ''

        self.initializer = embeddings_initializer if embeddings_initializer else EmbeddingInitializer()

    def build(self, input_shape):
        self.embedding_table = self.add_weight("embedding_table",
                                               shape=[self.input_dim, self.output_dim],
                                               dtype=self.dtype,
                                               initializer=self.initializer,
                                               trainable=self.trainable)

    def call(self, indices):
        return tf.gather(params=self.embedding_table, indices=indices)

    def save_checkpoint(self, checkpoint_path):
        filename = get_variable_path(checkpoint_path, self.feature_name)
        indices = tf.range(start=0, limit=self.embedding_table.shape[0], dtype=tf.int32)
        arr = tf.gather(params=self.embedding_table, indices=indices, axis=0)
        arr = arr.numpy()
        np.save(arr=arr, file=filename)

    def restore_checkpoint(self, checkpoint_path):
        filename = get_variable_path(checkpoint_path, self.feature_name)
        print('restoring embedding table from: ', filename)
        numpy_arr = np.load(file=filename, mmap_mode='r')

        num_chunks = math.ceil(numpy_arr.shape[0] / _embedding_checkpoint_batch)
        for i in range(num_chunks):
            begin = i * _embedding_checkpoint_batch
            end = (i+1) * _embedding_checkpoint_batch
            end = min(end, numpy_arr.shape[0])

            indices = tf.range(start=begin, limit=end, dtype=tf.int32)
            update = tf.IndexedSlices(values=numpy_arr[begin:end, :],
                                      indices=indices,
                                      dense_shape=self.embedding_table.shape)
            self.embedding_table.scatter_update(sparse_delta=update)


class EmbeddingGroup(tf.keras.layers.Layer):
    def __init__(self, table_sizes, output_dim, dtype=tf.float32, feature_names=None, trainable=True):
        super(EmbeddingGroup, self).__init__(dtype=dtype)
        self.table_sizes = table_sizes
        self.output_dim = output_dim
        self.feature_names = feature_names
        if not self.feature_names:
            self.feature_names = [f'feature_{i}' for i in range(len(table_sizes))]

        self.embedding_layers = []
        for fname, ts in zip(self.feature_names, self.table_sizes):
            self.embedding_layers.append(Embedding(ts, output_dim, dtype=self.dtype,
                                                   feature_name=fname, trainable=trainable))

    def call(self, indices):
        outputs = []
        for i, l in enumerate(self.embedding_layers):
            out = l(indices[:, i])
            out = tf.expand_dims(out, axis=1)
            outputs.append(out)
        result = tf.concat(outputs, axis=1)
        return result

    def save_checkpoint(self, checkpoint_path):
        for e in self.embedding_layers:
            e.save_checkpoint(checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        for e in self.embedding_layers:
            e.restore_checkpoint(checkpoint_path)


class FusedEmbeddingInitializer(tf.keras.initializers.Initializer):
    def __init__(self, table_sizes, embedding_dim, wrapped):
        self.table_sizes = table_sizes
        self.wrapped = wrapped
        self.embedding_dim = embedding_dim

    def __call__(self, shape, dtype=tf.float32):
        with tf.device('/CPU:0'):
            subtables = []
            for table_size in self.table_sizes:
                subtable = self.wrapped()(shape=[table_size, self.embedding_dim], dtype=dtype)
                subtables.append(subtable)
            weights = tf.concat(subtables, axis=0)
        return weights

    def get_config(self):
        return {}


class FusedEmbedding(tf.keras.layers.Layer):
    def __init__(self, table_sizes, output_dim, dtype=tf.float32, feature_names=None, trainable=True,
                 use_mde_embeddings=True):

        super(FusedEmbedding, self).__init__(dtype=dtype)
        self.table_sizes = table_sizes
        self.output_dim = output_dim
        self.offsets = np.array([0] + table_sizes, dtype=np.int32).cumsum()
        self.offsets.reshape([1, -1])
        self.offsets = tf.constant(self.offsets, dtype=tf.int32)
        self.use_mde_embeddings = use_mde_embeddings
        self.feature_names = feature_names
        if not self.feature_names:
            self.feature_names = [f'feature_{i}' for i in range(len(table_sizes))]
        self.trainable = trainable

        initializer = FusedEmbeddingInitializer(table_sizes=self.table_sizes,
                                                embedding_dim=self.output_dim,
                                                wrapped=EmbeddingInitializer)

        embedding_cls = embedding.Embedding if use_mde_embeddings else Embedding
        self.wrapped = embedding_cls(input_dim=self.offsets[-1], output_dim=self.output_dim,
                                     embeddings_initializer=initializer)

    def _get_embedding_table(self):
        if self.use_mde_embeddings:
            return self.wrapped.variables[0]
        else:
            return self.wrapped.variables[0]

    def call(self, indices):
        indices = indices + self.offsets[:-1]
        return self.wrapped(indices)

    def save_checkpoint(self, checkpoint_path):
        for j in range(len(self.offsets) - 1):
            nrows = self.offsets[j+1] - self.offsets[j]
            name = self.feature_names[j]
            filename = get_variable_path(checkpoint_path, name)

            indices = tf.range(start=self.offsets[j], limit=self.offsets[j] + nrows, dtype=tf.int32)
            arr = tf.gather(params=self._get_embedding_table(), indices=indices, axis=0)
            arr = arr.numpy()
            np.save(arr=arr, file=filename)

    def restore_checkpoint(self, checkpoint_path):
        for j in range(len(self.offsets) - 1):
            name = self.feature_names[j]

            filename = get_variable_path(checkpoint_path, name)
            print('restoring embedding table from: ', filename)
            numpy_arr = np.load(file=filename, mmap_mode='r')

            num_chunks = math.ceil(numpy_arr.shape[0] / _embedding_checkpoint_batch)
            for i in range(num_chunks):
                begin = i * _embedding_checkpoint_batch
                end = (i+1) * _embedding_checkpoint_batch
                end = min(end, numpy_arr.shape[0])

                indices = tf.range(start=begin, limit=end, dtype=tf.int32) + self.offsets[j]
                update = tf.IndexedSlices(values=numpy_arr[begin:end, :],
                                          indices=indices,
                                          dense_shape=self._get_embedding_table().shape)
                self._get_embedding_table().scatter_update(sparse_delta=update)


class DualEmbeddingGroup(tf.keras.layers.Layer):
    """
    A group of embeddings with the same output dimension.
    If it runs out of GPU memory it will use CPU memory for the largest tables.
    """

    def __init__(self, cardinalities, output_dim, memory_threshold,
                 cpu_embedding='multitable', gpu_embedding='fused', dtype=tf.float32,
                 feature_names=None, trainable=True, use_mde_embeddings=True):

        # TODO: throw an exception if the features are not sorted by cardinality in reversed order

        super(DualEmbeddingGroup, self).__init__(dtype=dtype)

        if dtype not in [tf.float32, tf.float16]:
            raise ValueError(f'Only float32 and float16 embedding dtypes are currently supported. Got {dtype}.')

        cpu_embedding_class = EmbeddingGroup if cpu_embedding == 'multitable' else FusedEmbedding
        gpu_embedding_class = EmbeddingGroup if gpu_embedding == 'multitable' else FusedEmbedding

        print('Dual embedding cardinalities: ', cardinalities)
        self.cardinalities = np.array(cardinalities)

        self.memory_threshold = memory_threshold

        self.bytes_per_element = 2 if self.dtype == tf.float16 else 4

        self.table_sizes = self.cardinalities * output_dim * self.bytes_per_element
        self._find_first_gpu_index()

        if not feature_names:
            feature_names = [f'feature_{i}' for i in range(len(self.table_sizes))]

        self.feature_names = feature_names

        self.gpu_embedding = gpu_embedding_class(table_sizes=self.gpu_cardinalities.tolist(),
                                                 output_dim=output_dim, dtype=self.dtype,
                                                 feature_names=[feature_names[i] for i in self.gpu_inputs],
                                                 trainable=trainable, use_mde_embeddings=use_mde_embeddings)

        # Force using FP32 for CPU embeddings, FP16 performance is much worse
        self.cpu_embedding = cpu_embedding_class(table_sizes=self.cpu_cardinalities,
                                                 output_dim=output_dim, dtype=tf.float32,
                                                 feature_names=[feature_names[i] for i in self.cpu_inputs],
                                                 trainable=trainable)

    def _find_first_gpu_index(self):
        # order from smallest to largest
        idx_mapping = np.argsort(self.table_sizes)
        reversed_sizes = self.table_sizes[idx_mapping]

        cumulative_size = np.cumsum(reversed_sizes)
        cumulative_indicators = (cumulative_size > self.memory_threshold * (10 ** 9)).tolist()
        if True in cumulative_indicators:
            index = cumulative_indicators.index(True)
        else:
            index = len(cumulative_size)

        self.first_cpu_index = index

        self.gpu_inputs = sorted(idx_mapping[:self.first_cpu_index])
        self.cpu_inputs = sorted(idx_mapping[self.first_cpu_index:])

        self.cpu_cardinalities = self.cardinalities[self.cpu_inputs]
        self.gpu_cardinalities = self.cardinalities[self.gpu_inputs]

        self.cpu_sizes = self.table_sizes[self.cpu_inputs]
        self.gpu_sizes = self.table_sizes[self.gpu_inputs]

        print(f'self.cpu_inputs: {self.cpu_inputs}')
        print(f'self.gpu_inputs: {self.gpu_inputs}')

        print(f'Total size of GPU tables: {sum(self.gpu_sizes) / 10 ** 9:.3f}[GB]')
        print(f'Total size of CPU tables: {sum(self.cpu_sizes) / 10 ** 9:.3f}[GB]')

    def call(self, indices):
        cpu_indices, gpu_indices = [], []

        if not self.cpu_inputs:
            return self.gpu_embedding(indices)

        if not self.gpu_inputs:
            with tf.device('/CPU:0'):
                return self.cpu_embedding(indices)

        for i in self.cpu_inputs:
            cpu_indices.append(indices[:, i])
        for i in self.gpu_inputs:
            gpu_indices.append(indices[:, i])

        to_concat = []
        # at least one cpu-based embedding
        with tf.device('/CPU:0'):
            cpu_indices = tf.stack(cpu_indices, axis=1)
            cpu_results = self.cpu_embedding(cpu_indices)
            cpu_results = tf.cast(cpu_results, dtype=self.dtype)
            to_concat.append(cpu_results)
        # at least one gpu-based embedding
        with tf.device('/GPU:0'):
            gpu_indices = tf.stack(gpu_indices, axis=1)
            gpu_results = self.gpu_embedding(gpu_indices)
            to_concat.append(gpu_results)

        result = tf.concat(to_concat, axis=1)

        reorder_indices = np.concatenate([self.cpu_inputs, self.gpu_inputs], axis=0).argsort().tolist()
        split_result = tf.split(result, num_or_size_splits=indices.shape[1], axis=1)
        result = [split_result[i] for i in reorder_indices]
        result = tf.concat(result, axis=1)
        return result

    def save_checkpoint(self, checkpoint_path):
        self.gpu_embedding.save_checkpoint(checkpoint_path)
        self.cpu_embedding.save_checkpoint(checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        self.gpu_embedding.restore_checkpoint(checkpoint_path)
        self.cpu_embedding.restore_checkpoint(checkpoint_path)
