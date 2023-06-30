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

import json

import tensorflow.keras.initializers as initializers
import math
from tensorflow.python.keras.saving.saving_utils import model_input_signature
from .dcn import CrossNetwork
from . import interaction
import tensorflow as tf
import horovod.tensorflow as hvd

try:
    from tensorflow_dot_based_interact.python.ops import dot_based_interact_ops
except ImportError:
    print('WARNING: Could not import the custom dot-interaction kernels')


dense_model_parameters = ['embedding_dim', 'interaction', 'bottom_mlp_dims',
                          'top_mlp_dims', 'num_numerical_features', 'categorical_cardinalities',
                          'transpose', 'num_cross_layers', 'cross_layer_projection_dim',
                          'batch_size']

class DenseModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(DenseModel, self).__init__()

        for field in dense_model_parameters:
            self.__dict__[field] = kwargs[field]

        self.num_all_categorical_features = len(self.categorical_cardinalities)
        self.bottom_mlp_dims = [int(d) for d in self.bottom_mlp_dims]
        self.top_mlp_dims = [int(d) for d in self.top_mlp_dims]

        if self.interaction != 'cross' and any(dim != self.embedding_dim[0] for dim in self.embedding_dim):
            raise ValueError(f'For DLRM all embedding dimensions should be equal, '
                             f'got interaction={interaction}, embedding_dim={self.embedding_dim}')

        if self.interaction != 'cross' and self.bottom_mlp_dims[-1] != self.embedding_dim[0]:
            raise ValueError(f'Final dimension of the Bottom MLP should match embedding dimension. '
                             f'Got: {self.bottom_mlp_dims[-1]} and {self.embedding_dim} respectively.')

        self._create_interaction_op()
        self._create_bottom_mlp()
        self._create_top_mlp()

        self.bottom_mlp_padding = self._compute_padding(num_features=self.num_numerical_features)
        self.top_mlp_padding = self._compute_padding(num_features=self._get_top_mlp_input_features())

    def _create_interaction_op(self):
        if self.interaction == 'dot_custom_cuda':
            self.interact_op = dot_based_interact_ops.dot_based_interact
        elif self.interaction == 'dot_tensorflow':
            # TODO: add support for datasets with no dense features
            self.interact_op = interaction.DotInteractionGather(num_features=self.num_all_categorical_features + 1)
        elif self.interaction == 'cross':
            self.interact_op = CrossNetwork(num_layers=self.num_cross_layers,
                                            projection_dim=self.cross_layer_projection_dim)
        else:
            raise ValueError(f'Unknown interaction {self.interaction}')

    @staticmethod
    def _compute_padding(num_features, multiple=8):
        pad_to = math.ceil(num_features / multiple) * multiple
        return pad_to - num_features

    def _get_top_mlp_input_features(self):
        if self.interaction == 'cross':
            num_features = sum(self.embedding_dim)
            if self.num_numerical_features != 0:
                num_features += self.bottom_mlp_dims[-1]
            return num_features
        else:
            num_features = self.num_all_categorical_features
            if self.num_numerical_features != 0:
                num_features += 1
            num_features = num_features * (num_features - 1)
            num_features = num_features // 2
            num_features = num_features + self.bottom_mlp_dims[-1]
            return num_features

    def _create_bottom_mlp(self):
        self.bottom_mlp_layers = []
        for dim in self.bottom_mlp_dims:
            kernel_initializer = initializers.GlorotNormal()
            bias_initializer = initializers.RandomNormal(stddev=math.sqrt(1. / dim))

            l = tf.keras.layers.Dense(dim, activation='relu',
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer)
            self.bottom_mlp_layers.append(l)

    def _create_top_mlp(self):
        self.top_mlp = []
        for i, dim in enumerate(self.top_mlp_dims):
            if i == len(self.top_mlp_dims) - 1:
                # final layer
                activation = 'linear'
            else:
                activation = 'relu'

            kernel_initializer = initializers.GlorotNormal()
            bias_initializer = initializers.RandomNormal(stddev=math.sqrt(1. / dim))

            l = tf.keras.layers.Dense(dim, activation=activation,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer)
            self.top_mlp.append(l)

    def transpose_nonequal_embedding_dim(self, embedding_outputs, numerical_features):
        # We get a table-major format here for inference,
        # but the sizes of the tables are not the same.
        # Therefore a simple transposition will not work,
        # we need to perform multiple splits and concats instead.

        # TODO: test this.
        embedding_outputs = tf.reshape(embedding_outputs, shape=[-1])
        batch_size = numerical_features.shape[0]
        split_sizes = [batch_size * dim for dim in self.embedding_dim]
        embedding_outputs = tf.split(embedding_outputs, num_or_size_splits=split_sizes)
        embedding_outputs = [tf.split(eout, num_or_size_splits=dim) for eout, dim in zip(embedding_outputs,
                                                                                         self.emdedding_dim)]
        transposed_outputs = [] * batch_size
        for i, o in enumerate(transposed_outputs):
            ith_sample = [out[i] for out in embedding_outputs]
            ith_sample = tf.concat(ith_sample, axis=1)
            transposed_outputs[i] = ith_sample
        transposed_outputs = tf.concat(transposed_outputs, axis=0)
        return tf.reshape(transposed_outputs, shape=[batch_size, sum(self.embedding_dim)])

    def transpose_input(self, embedding_outputs, numerical_features):
        if any(dim != self.embedding_dim[0] for dim in self.embedding_dim):
            return self.transpose_nonequal_embedding_dim(embedding_outputs, numerical_features)
        else:
            embedding_outputs = tf.reshape(embedding_outputs, shape=[self.num_all_categorical_features, -1, self.embedding_dim[0]])
            return tf.transpose(embedding_outputs, perm=[1, 0, 2])

    def reshape_input(self, embedding_outputs):
        if self.interaction == 'cross':
            return tf.reshape(embedding_outputs, shape=[-1, sum(self.embedding_dim)])
        else:
            return tf.reshape(embedding_outputs, shape=[-1, self.num_all_categorical_features, self.embedding_dim[0]])

    @tf.function
    def call(self, numerical_features, embedding_outputs, sigmoid=False, training=False):
        numerical_features = tf.reshape(numerical_features, shape=[-1, self.num_numerical_features])

        bottom_mlp_out = self._call_bottom_mlp(numerical_features, training)

        if self.transpose:
            embedding_outputs = self.transpose_input(embedding_outputs, numerical_features)
        embedding_outputs = self.reshape_input(embedding_outputs)

        x = self._call_interaction(embedding_outputs, bottom_mlp_out)
        x = self._call_top_mlp(x)

        if sigmoid:
            x = tf.math.sigmoid(x)

        x = tf.cast(x, tf.float32)
        return x

    def _pad_bottom_mlp_input(self, numerical_features, training):
        if training:
            # When training, padding with a statically fixed batch size so that XLA has better shape information.
            # This yields a significant (~15%) speedup for singleGPU DLRM.
            padding = tf.zeros(shape=[self.batch_size // hvd.size(), self.bottom_mlp_padding],
                               dtype=self.compute_dtype)
            x = tf.concat([numerical_features, padding], axis=1)
        else:
            # For inference, use tf.pad.
            # This way inference can be performed with any batch size on the deployed SavedModel.
            x = tf.pad(numerical_features, [[0, 0], [0, self.bottom_mlp_padding]])
        return x

    def _call_bottom_mlp(self, numerical_features, training):
        numerical_features = tf.cast(numerical_features, dtype=self.compute_dtype)

        x = self._pad_bottom_mlp_input(numerical_features, training)

        with tf.name_scope('bottom_mlp'):
            for l in self.bottom_mlp_layers:
                x = l(x)
            x = tf.expand_dims(x, axis=1)
            bottom_mlp_out = x
        return bottom_mlp_out

    def _call_interaction(self, embedding_outputs, bottom_mlp_out):
        if self.interaction == 'cross':
            bottom_mlp_out = tf.reshape(bottom_mlp_out, [-1, self.bottom_mlp_dims[-1]])
            x = tf.concat([bottom_mlp_out, embedding_outputs], axis=1)
            x = self.interact_op(x)
        else:
            bottom_part_output = tf.concat([bottom_mlp_out, embedding_outputs], axis=1)
            x = tf.reshape(bottom_part_output, shape=[-1, self.num_all_categorical_features + 1, self.embedding_dim[0]])
            bottom_mlp_out = tf.reshape(bottom_mlp_out, shape=[-1, self.bottom_mlp_dims[-1]])
            x = self.interact_op(x, bottom_mlp_out)
        return x

    def _call_top_mlp(self, x):
        if self.interaction != 'dot_custom_cuda':
            x = tf.reshape(x, [-1, self._get_top_mlp_input_features()])
            x = tf.pad(x, [[0, 0], [0, self.top_mlp_padding]])

        with tf.name_scope('top_mlp'):
            for i, l in enumerate(self.top_mlp):
                x = l(x)
        return x

    def save_model(self, path, save_input_signature=False):
        if save_input_signature:
            input_sig = model_input_signature(self, keep_original_batch_size=True)
            call_graph = tf.function(self)
            signatures = call_graph.get_concrete_function(input_sig[0])
        else:
            signatures = None

        tf.keras.models.save_model(model=self, filepath=path, overwrite=True, signatures=signatures)

    def force_initialization(self, batch_size=64, training=False, flattened_input=True):
        if flattened_input:
            embeddings_output = tf.zeros([batch_size * sum(self.embedding_dim)])
            numerical_input = tf.zeros([batch_size * self.num_numerical_features])
        else:
            embeddings_output = tf.zeros([batch_size, sum(self.embedding_dim)])
            numerical_input = tf.zeros([batch_size, self.num_numerical_features])

        _ = self(numerical_input, embeddings_output, sigmoid=False, training=training)


    @staticmethod
    def load_model(path):
        print('Loading a saved model from', path)

        loaded = tf.keras.models.load_model(path)
        return loaded

    def save_config(self, path):
        config = {k : self.__dict__[k] for k in dense_model_parameters}
        with open(path, 'w') as f:
            json.dump(obj=config, fp=f, indent=4)

    @staticmethod
    def from_config(path):
        with open(path) as f:
            config = json.load(fp=f)
        return DenseModel(**config)

