# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
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


import tensorflow as tf


def dot_interact(concat_features, bottom_mlp_out=None, skip_gather=False):
    # Interact features, select lower-triangular portion, and re-shape.
    interactions = tf.matmul(concat_features, concat_features, transpose_b=True)

    ones = tf.ones_like(interactions, dtype=tf.float32)
    upper_tri_mask = tf.linalg.band_part(ones, 0, -1)

    feature_dim = tf.shape(interactions)[-1]

    if skip_gather:
        upper_tri_bool = tf.cast(upper_tri_mask, tf.bool)
        activations = tf.where(
                condition=upper_tri_bool, x=tf.zeros_like(interactions), y=interactions)
        out_dim = feature_dim * feature_dim
    else:
        lower_tri_mask = ones - upper_tri_mask
        activations = tf.boolean_mask(interactions, lower_tri_mask)
        out_dim = feature_dim * (feature_dim - 1) // 2

    activations = tf.reshape(activations, shape=[-1, out_dim])

    if bottom_mlp_out is not None:
        bottom_mlp_out = tf.squeeze(bottom_mlp_out)
        activations = tf.concat([activations, bottom_mlp_out], axis=1)

    return activations


def dummy_dot_interact(concat_features, bottom_mlp_out=None):
    batch_size = tf.shape(concat_features)[0]
    num_features = tf.shape(concat_features)[1]
    concat_features = tf.math.reduce_mean(concat_features, axis=[2], keepdims=True)
    return dot_interact(concat_features, bottom_mlp_out)
