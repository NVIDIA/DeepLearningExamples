# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
import numpy as np


def power_law(k_min, k_max, alpha, x):
    """convert uniform distribution to power law distribution"""
    gamma = 1 - alpha
    y = pow(x * (pow(k_max, gamma) - pow(k_min, gamma)) + pow(k_min, gamma), 1.0 / gamma)
    return y.astype(np.int32)


def gen_power_law_data(batch_size, hotness, num_rows, alpha, variable_hotness):
    """naive power law distribution generator
    NOTE: Repetition is allowed in multi hot data.
    NOTE: The resulting values are sorted by frequency, that is, the index=0 is the most frequently occurring etc.
    """
    if variable_hotness:
        # at least one element fetched for each feature
        row_lengths = power_law(1, hotness, alpha, np.random.rand(batch_size))
        total_elements = np.sum(row_lengths)
        y = power_law(1, num_rows + 1, alpha, np.random.rand(total_elements)) - 1
        result = tf.RaggedTensor.from_row_lengths(values=y, row_lengths=row_lengths)
    else:
        y = power_law(1, num_rows + 1, alpha, np.random.rand(batch_size * hotness)) - 1
        row_lengths = tf.ones(shape=[batch_size], dtype=tf.int32) * hotness
        result = tf.RaggedTensor.from_row_lengths(values=y, row_lengths=row_lengths)
    return result


class SyntheticDataset:
    def __init__(self, batch_size, num_numerical_features, categorical_feature_cardinalities,
                 categorical_feature_hotness, categorical_feature_alpha, num_workers, variable_hotness=True,
                 constant=False, num_batches=int(1e9)):
        self.batch_size = batch_size
        self.num_numerical_features = num_numerical_features
        self.categorical_feature_cardinalities = categorical_feature_cardinalities
        self.categorical_feature_hotness = categorical_feature_hotness
        self.categorical_feature_alpha = categorical_feature_alpha
        self.variable_hotness = variable_hotness
        self.num_workers = num_workers
        self.num_batches = num_batches

        if len(categorical_feature_hotness) != len(categorical_feature_cardinalities):
            raise ValueError("DummyDataset mismatch between cardinalities and hotness lengths."
                             f"Got {len(categorical_feature_cardinalities)} cardinalities and "
                             f"{len(categorical_feature_hotness)} hotnesses")

        self.cat_features_count = len(
            categorical_feature_cardinalities) if categorical_feature_cardinalities is not None else 0
        self.num_features_count = num_numerical_features if num_numerical_features is not None else 0

        self.constant = constant
        if self.constant:
            (self.numerical_features, self.categorical_features), self.labels = self._generate()

    def _generate(self):

        numerical_features = tf.random.uniform(shape=[self.batch_size // self.num_workers, self.num_numerical_features],
                                                    dtype=tf.float32) if self.num_features_count else -1
        labels = tf.cast(tf.random.uniform(shape=[self.batch_size // self.num_workers, 1],
                                                maxval=2, dtype=tf.int32), tf.float32)

        categorical_features = []
        for cardinality, hotness, alpha in zip(self.categorical_feature_cardinalities,
                                               self.categorical_feature_hotness,
                                               self.categorical_feature_alpha):

            feature = gen_power_law_data(batch_size=self.batch_size, hotness=hotness,
                                         num_rows=cardinality, alpha=alpha,
                                         variable_hotness=self.variable_hotness)

            categorical_features.append(feature)
        return (numerical_features, categorical_features), labels

    def __next__(self):
        if self.constant:
            return (self.numerical_features, self.categorical_features), self.labels
        else:
            return self._generate()

    def __len__(self):
        return self.num_batches

    def op(self):
        return self

    def __iter__(self):
        return self

    def get_next(self):
        return self.__next__()


