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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from tensorflow_dot_based_interact.python.ops import dot_based_interact_ops
except ImportError:
  import dot_based_interact_ops

def dot_based_interact_native(input, bottom_mlp_output):
  # Dot Based Interact of the "input" tensor
  concat_features = tf.cast(input, tf.float32)
  interactions = tf.matmul(concat_features, concat_features, transpose_b=True)
  ones = tf.ones_like(interactions, dtype=concat_features.dtype)
  upper_tri_mask = tf.linalg.band_part(ones, 0, -1)
  feature_dim = tf.shape(interactions)[-1]
  lower_tri_mask = ones - upper_tri_mask
  activations = tf.boolean_mask(interactions, lower_tri_mask)
  out_dim = feature_dim * (feature_dim - 1) // 2
  activations = tf.reshape(activations, shape=[-1, out_dim])

  # Top Concatenation of the bottom_mlp_output with the interactions
  bottom_mlp_output = tf.cast(tf.squeeze(bottom_mlp_output, axis=1), tf.float32)
  top_concat = tf.concat([bottom_mlp_output, activations], axis=1)

  # Zero Padding for performance in upstream ops
  padding = tf.zeros([concat_features.shape[0], 1])
  zero_padded = tf.concat([top_concat, padding], axis=1)

  return zero_padded


class DotBasedInteractTest(test.TestCase):

  def input(self, batch_size, num_rows, num_cols, dtype):
    # Creates two random tensors to use as sample inputs to test with:
    # - input: With shape [batch_size, num_rows, num_cols]
    # - bottom_mlp_output: With shape [batch_size, 1, num_cols]
    # Where the first row of input is a copy of bottom_mlp_output
    mlp_rows = 1
    emb_rows = num_rows - mlp_rows
    bottom_mlp_output = tf.random.uniform(shape=[batch_size, mlp_rows, num_cols], dtype=dtype)
    embeddings = tf.random.uniform(shape=[batch_size, emb_rows, num_cols], dtype=dtype)
    input = tf.concat([bottom_mlp_output, embeddings], axis=1)
    return tf.Variable(input), tf.Variable(bottom_mlp_output)

  def forward(self, batch_size, num_rows, num_cols, dtype):
    with self.test_session() as sess:
      with ops.device("/gpu:0"):
        input, bottom_mlp_output = self.input(batch_size, num_rows, num_cols, dtype)
        expected = dot_based_interact_native(input, bottom_mlp_output)
        result = dot_based_interact_ops.dot_based_interact(input, bottom_mlp_output)
        return result, expected

  def backward(self, batch_size, num_rows, num_cols, dtype):
    with self.test_session() as sess:
      with ops.device("/gpu:0"):
        input, bottom_mlp_output = self.input(batch_size, num_rows, num_cols, dtype)
        with tf.GradientTape() as tape:
            output = dot_based_interact_native(input, bottom_mlp_output)
            expected = tape.gradient(output, [input, bottom_mlp_output])
        with tf.GradientTape() as tape:
            output = dot_based_interact_ops.dot_based_interact(input, bottom_mlp_output)
            result = tape.gradient(output, [input, bottom_mlp_output])
        return result[0], expected[0]

  def test_fp32(self):
    # Higher than normal tolerance on FP32 due to TF32 on Ampere
    self.assertAllClose(*self.forward(16, 32, 32, tf.float32), rtol=1e-03)

  def test_fp32_not_aligned(self):
    self.assertAllClose(*self.forward(17, 31, 37, tf.float32), rtol=1e-03)

  def test_grad_fp32(self):
    self.assertAllClose(*self.backward(16, 32, 32, tf.float32), rtol=1e-03)

  def test_grad_fp32_not_aligned(self):
    self.assertAllClose(*self.backward(17, 31, 37, tf.float32), rtol=1e-03)

  def test_fp16(self):
    self.assertAllCloseAccordingToType(*self.forward(16, 32, 32, tf.float16))

  def test_fp16_not_aligned(self):
    self.assertAllCloseAccordingToType(*self.forward(15, 31, 37, tf.float16))

  def test_grad_fp16(self):
    self.assertAllCloseAccordingToType(*self.backward(16, 32, 32, tf.float16))

  def test_grad_fp16_not_aligned(self):
    self.assertAllCloseAccordingToType(*self.backward(17, 31, 37, tf.float16))

if __name__ == '__main__':
  test.main()
