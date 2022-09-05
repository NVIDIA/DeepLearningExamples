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

import logging
import tensorflow as tf
from data.feature_spec import FeatureSpec
from trainer.model import layers as nvtlayers
from data.outbrain.defaults import NUMERICAL_CHANNEL, ONEHOT_CHANNEL, MULTIHOT_CHANNEL


def get_feature_columns(fspec: FeatureSpec, embedding_dimensions: dict, combiner):
    logger = logging.getLogger("tensorflow")
    wide_columns, deep_columns = [], []

    categorical_columns = fspec.get_names_by_channel(ONEHOT_CHANNEL) + fspec.get_names_by_channel(MULTIHOT_CHANNEL)
    cardinalities = fspec.get_cardinalities(features=categorical_columns)
    for column_name in categorical_columns:

        categorical_column = tf.feature_column.categorical_column_with_identity(
            column_name, num_buckets=cardinalities[column_name]
        )
        wrapped_column = tf.feature_column.embedding_column(
            categorical_column,
            dimension=embedding_dimensions[column_name],
            combiner=combiner,
        )

        wide_columns.append(categorical_column)
        deep_columns.append(wrapped_column)

    numerics = [
        tf.feature_column.numeric_column(column_name, shape=(1,), dtype=tf.float32)
        for column_name in fspec.get_names_by_channel(NUMERICAL_CHANNEL)
    ]

    wide_columns.extend(numerics)
    deep_columns.extend(numerics)

    logger.warning("deep columns: {}".format(len(deep_columns)))
    logger.warning("wide columns: {}".format(len(wide_columns)))
    logger.warning(
        "wide&deep intersection: {}".format(
            len(set(wide_columns).intersection(set(deep_columns)))
        )
    )
    return wide_columns, deep_columns

def get_input_features(feature_spec):
    features = {}

    numeric_columns = feature_spec.get_names_by_channel(NUMERICAL_CHANNEL)
    onehot_columns = feature_spec.get_names_by_channel(ONEHOT_CHANNEL)
    multihot_columns = feature_spec.get_names_by_channel(MULTIHOT_CHANNEL)

    # Numerical
    for feature in numeric_columns:
        features[feature] = tf.keras.Input(
            shape=(1,), batch_size=None, name=feature, dtype=tf.float32, sparse=False
        )

    # Categorical (One-hot)
    for feature in onehot_columns:
        features[feature] = tf.keras.Input(
            shape=(1,), batch_size=None, name=feature, dtype=tf.int32, sparse=False
        )

    # Categorical (Multi-hot)
    multihot_hotness_dict = feature_spec.get_multihot_hotnesses(multihot_columns)
    for feature, hotness in multihot_hotness_dict.items():
        features[feature] = tf.keras.Input(
            shape=(hotness,),
            batch_size=None,
            name=f"{feature}",
            dtype=tf.int32,
            sparse=False,
        )

    return features


def wide_deep_model(args, feature_spec, embedding_dimensions):
    wide_columns, deep_columns = get_feature_columns(fspec=feature_spec,
                                                     embedding_dimensions=embedding_dimensions,
                                                     combiner=args.combiner)
    features = get_input_features(feature_spec)

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