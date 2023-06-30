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
import time
import os

from utils.distributed import dist_print
from .dense_model import DenseModel, dense_model_parameters
from .sparse_model import SparseModel, sparse_model_parameters
from .nn_utils import create_inputs_dict


class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        if kwargs:
            dense_model_kwargs = {k:kwargs[k] for k in dense_model_parameters}
            self.dense_model = DenseModel(**dense_model_kwargs)

            sparse_model_kwargs = {k:kwargs[k] for k in sparse_model_parameters}
            self.sparse_model = SparseModel(**sparse_model_kwargs)

    @staticmethod
    def create_from_checkpoint(checkpoint_path):
        if checkpoint_path is None:
            return None

        model = Model()
        model.dense_model = DenseModel.from_config(os.path.join(checkpoint_path, 'dense', 'config.json'))
        model.sparse_model = SparseModel.from_config(os.path.join(checkpoint_path, 'sparse', 'config.json'))
        model.restore_checkpoint(checkpoint_path)
        return model

    def force_initialization(self, global_batch_size):
        numerical_features = tf.zeros(shape=[global_batch_size // hvd.size(),
                                      self.dense_model.num_numerical_features])

        categorical_features = [tf.zeros(shape=[global_batch_size, 1], dtype=tf.int32)
                                for _ in range(len(self.sparse_model.get_local_table_ids(hvd.rank())))]
        inputs = create_inputs_dict(numerical_features, categorical_features)
        self(inputs=inputs)

    @tf.function
    def call(self, inputs, sigmoid=False, training=False):
        numerical_features, cat_features = list(inputs.values())
        embedding_outputs = self.sparse_model(cat_features)
        embedding_outputs = tf.reshape(embedding_outputs, shape=[-1])
        x = self.dense_model(numerical_features, embedding_outputs, sigmoid=sigmoid, training=training)
        return x

    def save_checkpoint(self, checkpoint_path):
        dist_print('Saving a checkpoint...')
        begin_save = time.time()
        os.makedirs(checkpoint_path, exist_ok=True)

        if hvd.rank() == 0:
            dense_checkpoint_dir = os.path.join(checkpoint_path, 'dense')
            os.makedirs(dense_checkpoint_dir, exist_ok=True)
            self.dense_model.save_config(os.path.join(dense_checkpoint_dir, 'config.json'))
            self.dense_model.save_weights(os.path.join(dense_checkpoint_dir, 'dense'))

        sparse_checkpoint_dir = os.path.join(checkpoint_path, 'sparse')
        os.makedirs(sparse_checkpoint_dir, exist_ok=True)
        self.sparse_model.save_config(os.path.join(sparse_checkpoint_dir, 'config.json'))
        self.sparse_model.save_checkpoint(sparse_checkpoint_dir)

        end_save = time.time()
        dist_print('Saved a checkpoint to ', checkpoint_path)
        dist_print(f'Saving a checkpoint took {end_save - begin_save:.3f}')

    def restore_checkpoint(self, checkpoint_path):
        begin = time.time()
        dist_print('Restoring a checkpoint...')

        local_batch = 64
        self.force_initialization(global_batch_size=hvd.size()*local_batch)

        dense_checkpoint_path = os.path.join(checkpoint_path, 'dense', 'dense')
        self.dense_model.load_weights(dense_checkpoint_path)

        sparse_checkpoint_dir = os.path.join(checkpoint_path, 'sparse')
        self.sparse_model.load_checkpoint(sparse_checkpoint_dir)

        end = time.time()
        dist_print(f'Restoring a checkpoint took: {end-begin:.3f} seconds')
        return self
