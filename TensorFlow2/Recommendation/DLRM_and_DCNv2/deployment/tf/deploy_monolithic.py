# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import os

import tensorflow as tf
from tensorflow.python.saved_model import save_options

from nn.embedding import DualEmbeddingGroup
from nn.dense_model import DenseModel


class SparseModel(tf.keras.Model):
    def __init__(self, cardinalities, output_dim, memory_threshold):
        super().__init__()
        self.cardinalities = cardinalities
        self.output_dim = output_dim
        self.embedding = DualEmbeddingGroup(cardinalities, output_dim, memory_threshold, use_mde_embeddings=False)

    @tf.function
    def call(self, x):
        x = self.embedding(x)
        x = tf.reshape(x, [-1, len(self.cardinalities) * self.output_dim])
        return x


class Model(tf.keras.Model):
    def __init__(self, sparse_submodel, dense_submodel, cpu):
        super().__init__()
        self.sparse_submodel = sparse_submodel
        self.dense_submodel = dense_submodel
        self.cpu = cpu

    def call(self, numerical_features, cat_features):
        device = '/CPU:0' if self.cpu else '/GPU:0'
        with tf.device(device):
            embedding_outputs = self.sparse_submodel(cat_features)
            y = self.dense_submodel(numerical_features, embedding_outputs)
        return y

def load_dense(src, model_precision, model_format):
    dense_model = DenseModel.from_config(os.path.join(src, "config.json"))
    if dense_model.amp and model_precision == "fp16" and model_format == 'tf-savedmodel':
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    if dense_model.interaction == 'dot_custom_cuda':
        dense_model.interaction = 'dot_tensorflow'
        dense_model._create_interaction_op()

    dense_model.load_weights(os.path.join(src, "dense"))

    dense_model.transpose = False
    dense_model.force_initialization(training=False)
    return dense_model


def deploy_monolithic(
    sparse_src,
    dense_src,
    dst,
    model_name,
    max_batch_size,
    engine_count_per_device,
    num_gpus=1,
    version="1",
    cpu=False,
    model_precision='fp32'
):

    if model_precision == 'fp16':
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    dense_model = load_dense(src=dense_src, model_precision=model_precision, model_format='tf-savedmodel')

    print("deploy monolithic dst: ", dst)
    with open(os.path.join(sparse_src, "config.json")) as f:
        src_config = json.load(f)

    num_cat_features = len(src_config["categorical_cardinalities"])
    src_paths = [os.path.join(sparse_src, f"feature_{i}.npy") for i in range(num_cat_features)]

    sparse_model = SparseModel(cardinalities=src_config["categorical_cardinalities"],
                               output_dim=src_config['embedding_dim'][0],
                               memory_threshold=75 if not cpu else 0)

    model = Model(sparse_submodel=sparse_model, dense_submodel=dense_model, cpu=cpu)

    dummy_batch_size = 65536
    dummy_categorical = tf.zeros(shape=(dummy_batch_size, len(src_config["categorical_cardinalities"])), dtype=tf.int32)
    dummy_numerical = tf.zeros(shape=(dummy_batch_size, dense_model.num_numerical_features), dtype=tf.float32)

    _ = model(numerical_features=dummy_numerical, cat_features=dummy_categorical)

    options = save_options.SaveOptions(experimental_variable_policy=save_options.VariablePolicy.SAVE_VARIABLE_DEVICES)
    savedmodel_dir = os.path.join(dst, model_name, version, 'model.savedmodel')
    os.makedirs(savedmodel_dir)
    tf.keras.models.save_model(model=model, filepath=savedmodel_dir, overwrite=True, options=options)
