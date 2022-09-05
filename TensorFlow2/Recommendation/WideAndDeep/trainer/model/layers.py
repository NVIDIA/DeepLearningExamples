# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc


# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg


def _sort_columns(feature_columns):
    return sorted(feature_columns, key=lambda col: col.name)


def _validate_numeric_column(feature_column):
    if len(feature_column.shape) > 1:
        return (
            "Matrix numeric features are not allowed, "
            "found feature {} with shape {}".format(
                feature_column.key, feature_column.shape
            )
        )


def _validate_categorical_column(feature_column):
    if not isinstance(feature_column, fc.IdentityCategoricalColumn):
        return (
            "Only acceptable categorical columns for feeding "
            "embeddings are identity, found column {} of type {}. "
            "Consider using NVTabular online preprocessing to perform "
            "categorical transformations".format(
                feature_column.name, type(feature_column).__name__
            )
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
                    "Found bucketized column {}. DenseFeatures layer "
                    "cannot apply bucketization preprocessing. Consider using "
                    "NVTabular to do preprocessing offline".format(feature_column.name)
                )
        elif isinstance(feature_column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            _errors.append(
                _validate_categorical_column(feature_column.categorical_column)
            )

        elif isinstance(feature_column, fc.NumericColumn):
            _errors.append(_validate_numeric_column(feature_column))

    _errors = list(filter(lambda e: e is not None, _errors))
    if len(_errors) > 0:
        msg = "Found issues with columns passed to DenseFeatures:"
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
            "embeddings and continuous features to have same "
            "size. Found dimensions {}".format(dims)
        )


def _categorical_embedding_lookup(table, inputs, feature_name, combiner):
    # Multi-hots
    if inputs[feature_name].shape[1] > 1:

        # Multi-hot embedding lookup
        x = inputs[feature_name]
        embeddings = tf.gather(table, x)

        # Remove padded values
        # This is an inverse of dataloader pad_batch
        mask_array = tf.cast(x >= 0, embeddings.dtype)
        mask = tf.expand_dims(mask_array, -1)
        embeddings = tf.math.multiply(embeddings, mask)

        # Sum aggregation
        embeddings = tf.reduce_sum(embeddings, axis=1)

        # Divide by number of not zeros if mean aggregation
        if combiner == "mean":
            row_lengths = tf.reduce_sum(mask_array, axis=1)
            row_lengths = tf.cast(row_lengths, embeddings.dtype)
            row_lengths = tf.expand_dims(row_lengths, -1)
            embeddings = tf.math.divide_no_nan(embeddings, row_lengths)
    else:
        embeddings = tf.gather(table, inputs[feature_name][:, 0])

    return embeddings


def _handle_continuous_feature(inputs, feature_column):
    if feature_column.shape[0] > 1:
        x = inputs[feature_column.name]
        if isinstance(x, tuple):
            x = x[0]
        return tf.reshape(x, (-1, feature_column.shape[0]))
    return inputs[feature_column.name]


class DenseFeatures(tf.keras.layers.Layer):
    """
    Layer which maps a dictionary of input tensors to a dense, continuous
    vector digestible by a neural network. Meant to reproduce the API exposed
    by `tf.keras.layers.DenseFeatures` while reducing overhead for the
    case of one-hot categorical and scalar numeric features.
    Uses TensorFlow `feature_column` to represent inputs to the layer, but
    does not perform any preprocessing associated with those columns. As such,
    it should only be passed `numeric_column` objects and their subclasses,
    `embedding_column` and `indicator_column`. Preprocessing functionality should
    be moved to NVTabular.
    For multi-hot categorical or vector continuous data, represent the data for
    a feature with a dictionary entry `"<feature_name>__values"` corresponding
    to the flattened array of all values in the batch. For multi-hot categorical
    data, there should be a corresponding `"<feature_name>__nnzs"` entry that
    describes how many categories are present in each sample (and so has length
    `batch_size`).
    Note that categorical columns should be wrapped in embedding or
    indicator columns first, consistent with the API used by
    `tf.keras.layers.DenseFeatures`.
    Example usage::
        column_a = tf.feature_column.numeric_column("a", (1,))
        column_b = tf.feature_column.categorical_column_with_identity("b", 100)
        column_b_embedding = tf.feature_column.embedding_column(column_b, 4)
        inputs = {
            "a": tf.keras.Input(name="a", shape=(1,), dtype=tf.float32),
            "b": tf.keras.Input(name="b", shape=(1,), dtype=tf.int64)
        }
        x = DenseFeatures([column_a, column_b_embedding])(inputs)
    Parameters
    ----------
    feature_columns : list of `tf.feature_column`
        feature columns describing the inputs to the layer
    aggregation : str in ("concat", "stack")
        how to combine the embeddings from multiple features
    """

    def __init__(self, feature_columns, aggregation="concat", name=None, **kwargs):
        # sort feature columns to make layer independent of column order
        feature_columns = _sort_columns(feature_columns)
        _validate_dense_feature_columns(feature_columns)

        if aggregation == "stack":
            _validate_stack_dimensions(feature_columns)
        elif aggregation != "concat":
            raise ValueError(
                "Unrecognized aggregation {}, must be stack or concat".format(
                    aggregation
                )
            )

        self.feature_columns = feature_columns
        self.aggregation = aggregation
        super(DenseFeatures, self).__init__(name=name, **kwargs)

    def build(self, input_shapes):
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
                x = _handle_continuous_feature(inputs, feature_column)
                features.append(x)
            else:
                feature_name = feature_column.categorical_column.name
                table = self.embedding_tables[feature_name]
                combiner = getattr(feature_column, "combiner", "sum")
                embeddings = _categorical_embedding_lookup(
                    table, inputs, feature_name, combiner
                )
                features.append(embeddings)

        if self.aggregation == "stack":
            return tf.stack(features, axis=1)
        return tf.concat(features, axis=1)

    def compute_output_shape(self, input_shapes):
        input_shape = list(input_shapes.values())[0]
        if self.aggregation == "concat":
            output_dim = len(self.numeric_features) + sum(
                [shape[-1] for shape in self.embedding_shapes.values()]
            )
            return (input_shape[0], output_dim)
        else:
            embedding_dim = list(self.embedding_shapes.values())[0]
            return (input_shape[0], len(self.embedding_shapes), embedding_dim)

    def get_config(self):
        return {
            "feature_columns": self.feature_columns,
            "aggregation": self.aggregation,
        }


def _validate_linear_feature_columns(feature_columns):
    _errors = []
    for feature_column in feature_columns:
        if isinstance(feature_column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            _errors.append(
                "Only pass categorical or numeric columns to ScalarLinearFeatures "
                "layer, found column {} of type".format(feature_column)
            )
        elif isinstance(feature_column, fc.NumericColumn):
            _errors.append(_validate_numeric_column(feature_column))
        else:
            _errors.append(_validate_categorical_column(feature_column))

    _errors = list(filter(lambda e: e is not None, _errors))
    if len(_errors) > 0:
        msg = "Found issues with columns passed to ScalarDenseFeatures:"
        msg += "\n\t".join(_errors)
        raise ValueError(_errors)


# TODO: is there a clean way to combine these two layers
# into one, maybe with a "sum" aggregation? Major differences
# seem to be whether categorical columns are wrapped in
# embeddings and the numeric matmul, both of which seem
# reasonably easy to check. At the very least, we should
# be able to subclass I think?
class LinearFeatures(tf.keras.layers.Layer):
    """
    Layer which implements a linear combination of one-hot categorical
    and scalar numeric features. Based on the "wide" branch of the Wide & Deep
    network architecture.
    Uses TensorFlow ``feature_column``s to represent inputs to the layer, but
    does not perform any preprocessing associated with those columns. As such,
    it should only be passed ``numeric_column`` and
    ``categorical_column_with_identity``. Preprocessing functionality should
    be moved to NVTabular.
    Also note that, unlike ScalarDenseFeatures, categorical columns should
    NOT be wrapped in embedding or indicator columns first.
    Example usage::
        column_a = tf.feature_column.numeric_column("a", (1,))
        column_b = tf.feature_column.categorical_column_with_identity("b", 100)
        inputs = {
            "a": tf.keras.Input(name="a", shape=(1,), dtype=tf.float32),
            "b": tf.keras.Input(name="b", shape=(1,), dtype=tf.int64)
        }
        x = ScalarLinearFeatures([column_a, column_b])(inputs)
    Parameters
    ----------
    feature_columns : list of tf.feature_column
        feature columns describing the inputs to the layer
    """

    def __init__(self, feature_columns, name=None, **kwargs):
        feature_columns = _sort_columns(feature_columns)
        _validate_linear_feature_columns(feature_columns)

        self.feature_columns = feature_columns
        super(LinearFeatures, self).__init__(name=name, **kwargs)

    def build(self, input_shapes):
        # TODO: I've tried combining all the categorical tables
        # into a single giant lookup op, but it ends up turning
        # out the adding the offsets to lookup indices at call
        # time ends up being much slower due to kernel overhead
        # Still, a better (and probably custom) solutions would
        # probably be desirable
        numeric_kernel_dim = 0
        self.embedding_tables = {}
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                numeric_kernel_dim += feature_column.shape[0]
                continue

            self.embedding_tables[feature_column.key] = self.add_weight(
                name="{}/embedding_weights".format(feature_column.key),
                initializer="zeros",
                trainable=True,
                shape=(feature_column.num_buckets, 1),
            )
        if numeric_kernel_dim > 0:
            self.embedding_tables["numeric"] = self.add_weight(
                name="numeric/embedding_weights",
                initializer="zeros",
                trainable=True,
                shape=(numeric_kernel_dim, 1),
            )

        self.bias = self.add_weight(
            name="bias", initializer="zeros", trainable=True, shape=(1,)
        )
        self.built = True

    def call(self, inputs):
        x = self.bias
        numeric_inputs = []
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                numeric_inputs.append(
                    _handle_continuous_feature(inputs, feature_column)
                )
            else:
                table = self.embedding_tables[feature_column.key]
                embeddings = _categorical_embedding_lookup(
                    table, inputs, feature_column.key, "sum"
                )
                x = x + embeddings

        if len(numeric_inputs) > 0:
            numerics = tf.concat(numeric_inputs, axis=1)
            x = x + tf.matmul(numerics, self.embedding_tables["numeric"])
        return x

    def compute_output_shape(self, input_shapes):
        batch_size = list(input_shapes.values())[0].shape[0]
        return (batch_size, 1)

    def get_config(self):
        return {
            "feature_columns": self.feature_columns,
        }
