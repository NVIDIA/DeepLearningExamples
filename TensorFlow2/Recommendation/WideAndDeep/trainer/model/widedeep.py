# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
from data.outbrain.features import (CATEGORICAL_COLUMNS, MULTIHOT_COLUMNS,
                                    NUMERIC_COLUMNS, ONEHOT_COLUMNS,
                                    get_feature_columns)

from trainer.model import layers as nvtlayers


def get_inputs_columns(combiner):
    wide_columns, deep_columns = get_feature_columns(combiner)

    features = {}

    # Numerical
    for feature in NUMERIC_COLUMNS:
        features[feature] = tf.keras.Input(
            shape=(1,), batch_size=None, name=feature, dtype=tf.float32, sparse=False
        )

    # Categorical (One-hot)
    for feature in ONEHOT_COLUMNS:
        features[feature] = tf.keras.Input(
            shape=(1,), batch_size=None, name=feature, dtype=tf.int32, sparse=False
        )

    # Categorical (Multi-hot)
    for feature, hotness in MULTIHOT_COLUMNS.items():
        features[feature] = tf.keras.Input(
            shape=(hotness,),
            batch_size=None,
            name=f"{feature}",
            dtype=tf.int32,
            sparse=False,
        )

    return deep_columns, wide_columns, features


def wide_deep_model(args):
    deep_columns, wide_columns, features = get_inputs_columns(combiner=args.combiner)

    wide = nvtlayers.LinearFeatures(wide_columns, name="wide_linear")(features)

    dnn = nvtlayers.DenseFeatures(deep_columns, name="deep_embedded")(features)
    for unit_size in args.deep_hidden_units:
        dnn = tf.keras.layers.Dense(units=unit_size, activation="relu")(dnn)
        dnn = tf.keras.layers.Dropout(rate=args.deep_dropout)(dnn)
    dnn = tf.keras.layers.Dense(units=1)(dnn)

    dnn_model = tf.keras.Model(inputs=features, outputs=dnn)
    linear_model = tf.keras.Model(inputs=features, outputs=wide)

    model = tf.keras.experimental.WideDeepModel(
        linear_model, dnn_model, activation="sigmoid"
    )

    return model, features


def get_dummy_inputs(batch_size):
    inputs = {}
    shape = (batch_size, 1)
    for cat in CATEGORICAL_COLUMNS:
        inputs[cat] = tf.zeros(shape, dtype=tf.dtypes.int32)

    for num in NUMERIC_COLUMNS:
        inputs[num] = tf.zeros(shape, dtype=tf.dtypes.float32)

    for mul, hotness in MULTIHOT_COLUMNS.items():
        inputs[mul] = tf.zeros((batch_size, hotness), dtype=tf.dtypes.int32)

    labels = tf.zeros(shape, dtype=tf.dtypes.int32)

    return inputs, labels
