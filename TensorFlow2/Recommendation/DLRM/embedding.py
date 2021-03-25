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
import math


class EmbeddingInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=tf.float32):
        with tf.device('/CPU:0'):
            maxval = tf.sqrt(tf.constant(1.) / tf.cast(shape[0], tf.float32))
            maxval = tf.cast(maxval, dtype=dtype)
            minval = -maxval

            weights = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
            weights = tf.cast(weights, dtype=tf.float32)
        return weights

    def get_config(self):
        return {}


def _divisors(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield int(divisor)


def _get_n_chunks(input_dim, output_dim, max_chunk_size):
    for n_chunks in _divisors(output_dim):
        chunk_output_dim = output_dim / n_chunks
        chunk_size = input_dim * chunk_output_dim
        if chunk_size < max_chunk_size:
            return n_chunks
    raise ValueError(f'Unable to split embedding table: [{input_dim}, {output_dim}]')


class SplitEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, trainable=True, max_chunk_size=2**31):
        super(SplitEmbedding, self).__init__(dtype=tf.float32)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_tables = []
        self.trainable = trainable

        self.n_chunks = _get_n_chunks(input_dim, output_dim, max_chunk_size)

        self.chunk_output_dim = self.output_dim // self.n_chunks

        if self.n_chunks > output_dim:
            raise ValueError('Unable to perform a column-wise split of an embedding table!')

        if self.n_chunks > 1:
            print(f'Splitting the embedding table: [{input_dim} x {output_dim} into {self.n_chunks}'
                  f' [{input_dim} x {self.chunk_output_dim}] chunks')

    def build(self, input_shape):
        for i in range(self.n_chunks):
            w = self.add_weight(f"embedding_table_chunk_{i}",
                                shape=[self.input_dim, self.chunk_output_dim],
                                dtype=tf.float32,
                                initializer=EmbeddingInitializer(),
                                trainable=self.trainable,
                                )
            self.embedding_tables.append(w)

    def call(self, indices):
        outputs = []
        for embedding_table in self.embedding_tables:
            out = tf.gather(params=embedding_table, indices=indices)
            outputs.append(out)
        return tf.concat(outputs, axis=2)


class Embedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, trainable=True):
        super(Embedding, self).__init__(dtype=tf.float32)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_table = None
        self.trainable = trainable

    def build(self, input_shape):
        self.embedding_table = self.add_weight("embedding_table",
                                               shape=[self.input_dim, self.output_dim],
                                               dtype=tf.float32,
                                               initializer=EmbeddingInitializer(),
                                               trainable=self.trainable,
                                               )

    def call(self, indices):
        return tf.gather(params=self.embedding_table, indices=indices)
