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
import numpy as np
import math

from utils import get_variable_path


# write embedding checkpoints of 1M rows at a time
_embedding_checkpoint_batch = 1024 * 1024


@tf.keras.utils.register_keras_serializable()
class EmbeddingInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=tf.float32):
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
    def __init__(self, input_dim, output_dim, trainable=True, dtype=tf.float32, feature_name=None):
        super(Embedding, self).__init__(dtype=dtype)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_table = None
        self.trainable = trainable

        self.feature_name = feature_name
        if not self.feature_name:
            self.feature_name = ''

    def build(self, input_shape):
        self.embedding_table = self.add_weight("embedding_table",
                                               shape=[self.input_dim, self.output_dim],
                                               dtype=self.dtype,
                                               initializer=EmbeddingInitializer(),
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
        numpy_arr = np.load(file=filename)
        indices = tf.range(start=0, limit=numpy_arr.shape[0], dtype=tf.int32)
        update = tf.IndexedSlices(values=numpy_arr, indices=indices, dense_shape=self.embedding_table.shape)
        self.embedding_table.scatter_update(sparse_delta=update)


class EmbeddingGroup(tf.keras.layers.Layer):
    def __init__(self, table_sizes, output_dim, dtype=tf.float32, feature_names=None, trainable=True):
        super(EmbeddingGroup, self).__init__(dtype=dtype)
        self.table_sizes = table_sizes
        self.output_dim = output_dim
        self.feature_names = feature_names
        if not self.feature_names:
            self.feature_names = ['feature_{i}' for i in range(len(table_sizes))]

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


class JointEmbeddingInitializer(tf.keras.initializers.Initializer):
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


class JointEmbedding(tf.keras.layers.Layer):
    def __init__(self, table_sizes, output_dim, dtype, feature_names=None, trainable=True):
        super(JointEmbedding, self).__init__(dtype=dtype)
        self.table_sizes = table_sizes
        self.output_dim = output_dim
        self.embedding_table = None
        self.offsets = np.array([0] + table_sizes, dtype=np.int32).cumsum()
        self.offsets.reshape([1, -1])
        self.offsets = tf.constant(self.offsets, dtype=tf.int32)
        self.feature_names = feature_names
        if not self.feature_names:
            self.feature_names = ['feature_{i}' for i in range(len(table_sizes))]
        self.trainable = trainable

    def build(self, input_shape):
        initializer = JointEmbeddingInitializer(table_sizes=self.table_sizes,
                                                embedding_dim=self.output_dim,
                                                wrapped=EmbeddingInitializer)

        self.embedding_table = self.add_weight("embedding_table",
                                               shape=[self.offsets[-1], self.output_dim],
                                               dtype=self.dtype,
                                               initializer=initializer,
                                               trainable=self.trainable)

    def call(self, indices):
        indices = indices + self.offsets[:-1]
        return tf.nn.embedding_lookup(params=self.embedding_table, ids=indices)

    def save_checkpoint(self, checkpoint_path):
        for j in range(len(self.offsets) - 1):
            nrows = self.offsets[j+1] - self.offsets[j]
            name = self.feature_names[j]
            filename = get_variable_path(checkpoint_path, name)

            indices = tf.range(start=self.offsets[j], limit=self.offsets[j] + nrows, dtype=tf.int32)
            arr = tf.gather(params=self.embedding_table, indices=indices, axis=0)
            arr = arr.numpy()
            np.save(arr=arr, file=filename)

    def restore_checkpoint(self, checkpoint_path):
        for j in range(len(self.offsets) - 1):
            name = self.feature_names[j]

            filename = get_variable_path(checkpoint_path, name)
            numpy_arr = np.load(file=filename)

            indices = tf.range(start=self.offsets[j], limit=self.offsets[j] + numpy_arr.shape[0], dtype=tf.int32)
            update = tf.IndexedSlices(values=numpy_arr, indices=indices, dense_shape=self.embedding_table.shape)
            self.embedding_table.scatter_update(sparse_delta=update)


class DualEmbeddingGroup(tf.keras.layers.Layer):
    """
    A group of embeddings with the same output dimension.
    If it runs out of GPU memory it will use CPU memory for the largest tables.
    """

    def __init__(self, cardinalities, output_dim, memory_threshold,
                 cpu_embedding='multitable', gpu_embedding='fused', dtype=tf.float32,
                 feature_names=None, trainable=True):

        # TODO: throw an exception if the features are not sorted by cardinality in reversed order

        super(DualEmbeddingGroup, self).__init__(dtype=dtype)

        if dtype not in [tf.float32, tf.float16]:
            raise ValueError(f'Only float32 and float16 embedding dtypes are currently supported. Got {dtype}.')

        cpu_embedding_class = EmbeddingGroup if cpu_embedding == 'multitable' else JointEmbedding
        gpu_embedding_class = EmbeddingGroup if gpu_embedding == 'multitable' else JointEmbedding

        cardinalities = np.array(cardinalities)

        self.memory_threshold = memory_threshold

        self.bytes_per_element = 2 if self.dtype == tf.float16 else 4

        self.table_sizes = cardinalities * output_dim * self.bytes_per_element
        self._find_first_gpu_index()
        self.cpu_cardinalities = cardinalities[:self.first_gpu_index]
        self.gpu_cardinalities = cardinalities[self.first_gpu_index:]

        if not feature_names:
            feature_names = [f'feature_{i}' for i in range(len(self.table_sizes))]

        self.feature_names = feature_names

        self.gpu_embedding = gpu_embedding_class(table_sizes=self.gpu_cardinalities.tolist(),
                                                 output_dim=output_dim, dtype=self.dtype,
                                                 feature_names=feature_names[self.first_gpu_index:],
                                                 trainable=trainable)

        # Force using FP32 for CPU embeddings, FP16 performance is much worse
        self.cpu_embeddings = cpu_embedding_class(table_sizes=self.cpu_cardinalities,
                                                  output_dim=output_dim, dtype=tf.float32,
                                                  feature_names=feature_names[:self.first_gpu_index],
                                                  trainable=trainable)

    def _find_first_gpu_index(self):
        # order from smallest to largest
        reversed_sizes = np.flip(self.table_sizes)
        cumulative_size = np.cumsum(reversed_sizes)
        cumulative_indicators = (cumulative_size > self.memory_threshold * 2 ** 30).tolist()
        if True in cumulative_indicators:
            reversed_index = cumulative_indicators.index(True)
        else:
            reversed_index = len(cumulative_size)

        # convert to index into the original unreversed order
        index = len(reversed_sizes) - reversed_index
        self.first_gpu_index = index

    def call(self, indices):
        indices = tf.stack(indices, axis=1)

        to_concat = []
        if self.first_gpu_index > 0:
            # at least one cpu-based embedding
            cpu_indices = indices[:, :self.first_gpu_index]
            with tf.device('/CPU:0'):
                cpu_results = self.cpu_embeddings(cpu_indices)
                cpu_results = tf.cast(cpu_results, dtype=self.dtype)
                to_concat.append(cpu_results)

        if self.first_gpu_index < len(self.table_sizes):
            # at least one gpu-based embedding
            gpu_indices = indices[:, self.first_gpu_index:]
            gpu_results = self.gpu_embedding(gpu_indices)
            to_concat.append(gpu_results)

        if len(to_concat) > 1:
            result = tf.concat(to_concat, axis=1)
        else:
            result = to_concat[0]
        return result

    def save_checkpoint(self, checkpoint_path):
        self.gpu_embedding.save_checkpoint(checkpoint_path)
        self.cpu_embeddings.save_checkpoint(checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        self.gpu_embedding.restore_checkpoint(checkpoint_path)
        self.cpu_embeddings.restore_checkpoint(checkpoint_path)
