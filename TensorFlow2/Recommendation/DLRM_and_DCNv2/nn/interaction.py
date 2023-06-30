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


class DotInteractionGather(tf.keras.layers.Layer):
    def __init__(self, num_features):
        super(DotInteractionGather, self).__init__()
        self.num_features = num_features
        self.indices = []
        for i in range(self.num_features):
            for j in range(i):
                self.indices.append(i * num_features + j)

    def call(self, features, bottom_mlp_out=None):
        interactions = tf.matmul(features, features, transpose_b=True)
        interactions = tf.reshape(interactions, shape=[-1, self.num_features * self.num_features])

        x = tf.gather(params=interactions, indices=self.indices, axis=1)

        if bottom_mlp_out is not None:
            x = tf.concat([bottom_mlp_out, x], axis=1)
        return x