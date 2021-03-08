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

import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc


def _sort_columns(feature_columns):
    return sorted(feature_columns, key=lambda col: col.name)


def _validate_numeric_column(feature_column):
    if len(feature_column.shape) > 1:
        return "Matrix numeric utils are not allowed, " "found feature {} with shape {}".format(
            feature_column.key, feature_column.shape
        )
    elif feature_column.shape[0] != 1:
        return "Vector numeric utils are not allowed, " "found feature {} with shape {}".format(
            feature_column.key, feature_column.shape
        )


def _validate_categorical_column(feature_column):
    if not isinstance(feature_column, fc.IdentityCategoricalColumn):
        return (
            "Only acceptable categorical columns for feeding "
            "embeddings are identity, found column {} of type {}. "
            "Consider using NVTabular online preprocessing to perform "
            "categorical transformations".format(feature_column.name, type(feature_column).__name__)
        )


def _validate_dense_feature_columns(feature_columns):
    _errors = []
    for feature_column in feature_columns:
        if isinstance(feature_column, fc.CategoricalColumn):
            if not isinstance(feature_column, fc.BucketizedColumn):
                _errors.append(
                    "All feature columns must be dense, found categorical "
                    "column {} of type {}. Please wrap categorical columns "
                    "in embedding or indicator columns before passing".format(
                        feature_column.name, type(feature_column).__name__
                    )
                )
            else:
                _errors.append(
                    "Found bucketized column {}. ScalarDenseFeatures layer "
                    "cannot apply bucketization preprocessing. Consider using "
                    "NVTabular to do preprocessing offline".format(feature_column.name)
                )
        elif isinstance(feature_column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            _errors.append(_validate_categorical_column(feature_column.categorical_column))

        elif isinstance(feature_column, fc.NumericColumn):
            _errors.append(_validate_numeric_column(feature_column))

    _errors = list(filter(lambda e: e is not None, _errors))
    if len(_errors) > 0:
        msg = "Found issues with columns passed to ScalarDenseFeatures:"
        msg += "\n\t".join(_errors)
        raise ValueError(_errors)


def _validate_stack_dimensions(feature_columns):
    dims = []
    for feature_column in feature_columns:
        if isinstance(feature_column, fc.EmbeddingColumn):
            dimension = feature_column.dimension
        elif isinstance(feature_column, fc.IndicatorColumn):
            dimension = feature_column.categorical_column.num_buckets
        else:
            dimension = feature_column.shape[0]

        dims.append(dimension)

    dim0 = dims[0]
    if not all(dim == dim0 for dim in dims[1:]):
        dims = ", ".join(map(str, dims))
        raise ValueError(
            "'stack' aggregation requires all categorical "
            "embeddings and continuous utils to have same "
            "size. Found dimensions {}".format(dims)
        )


class ScalarDenseFeatures(tf.keras.layers.Layer):
    def __init__(self, feature_columns, aggregation="concat", name=None, **kwargs):
        feature_columns = _sort_columns(feature_columns)
        _validate_dense_feature_columns(feature_columns)

        assert aggregation in ("concat", "stack")
        if aggregation == "stack":
            _validate_stack_dimensions(feature_columns)

        self.feature_columns = feature_columns
        self.aggregation = aggregation
        super(ScalarDenseFeatures, self).__init__(name=name, **kwargs)

    def build(self, input_shapes):
        assert all(shape[1] == 1 for shape in input_shapes.values())

        self.embedding_tables = {}
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                continue

            feature_name = feature_column.categorical_column.key
            num_buckets = feature_column.categorical_column.num_buckets
            if isinstance(feature_column, fc.EmbeddingColumn):
                self.embedding_tables[feature_name] = self.add_weight(
                    name="{}/embedding_weights".format(feature_name),
                    trainable=True,
                    initializer="glorot_normal",
                    shape=(num_buckets, feature_column.dimension),
                )
            else:
                self.embedding_tables[feature_name] = self.add_weight(
                    name="{}/embedding_weights".format(feature_name),
                    trainable=False,
                    initializer=tf.constant_initializer(np.eye(num_buckets)),
                    shape=(num_buckets, num_buckets),
                )
        self.built = True

    def call(self, inputs):
        features = []
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                features.append(inputs[feature_column.name])
            else:
                feature_name = feature_column.categorical_column.name
                table = self.embedding_tables[feature_name]
                embeddings = tf.gather(table, inputs[feature_name][:, 0])
                features.append(embeddings)

        if self.aggregation == "stack":
            return tf.stack(features, axis=1)
        return tf.concat(features, axis=1)

    def compute_output_shape(self, input_shapes):
        input_shape = [i for i in input_shapes.values()][0]
        if self.aggregation == "concat":
            output_dim = len(self.numeric_features) + sum(
                [shape[-1] for shape in self.embedding_shapes.values()]
            )
            return (input_shape[0], output_dim)
        else:
            embedding_dim = [i for i in self.embedding_shapes.values()][0]
            return (input_shape[0], len(self.embedding_shapes), embedding_dim)

    def get_config(self):
        return {
            "feature_columns": self.feature_columns,
            "aggregation": self.aggregation,
        }
