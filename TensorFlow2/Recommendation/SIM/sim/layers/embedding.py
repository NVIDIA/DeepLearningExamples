# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


class EmbeddingInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=tf.float32):
        maxval = tf.sqrt(tf.constant(1.) / tf.cast(shape[0], tf.float32))
        maxval = tf.cast(maxval, dtype=dtype)
        minval = -maxval

        weights = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
        weights = tf.cast(weights, dtype=tf.float32)
        return weights

    def get_config(self):
        return {}


# https://github.com/NVIDIA/DeepLearningExamples/blob/81ee705868a11d6fe18c12d237abe4a08aab5fd6/TensorFlow2/Recommendation/DLRM/embedding.py#L94
class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        *,
        trainable=True,
        embedding_name=None,
        initializer=EmbeddingInitializer()
    ):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_name = (
            embedding_name if embedding_name is not None else "embedding_table"
        )
        self.embedding_table = None
        self.trainable = trainable
        self.initializer = initializer

    def build(self, input_shape):
        self.embedding_table = self.add_weight(
            self.embedding_name,
            shape=[self.input_dim, self.output_dim],
            dtype=tf.float32,
            initializer=self.initializer,
            trainable=self.trainable,
        )

    @tf.function
    def call(self, indices):
        return tf.gather(params=self.embedding_table, indices=indices)
