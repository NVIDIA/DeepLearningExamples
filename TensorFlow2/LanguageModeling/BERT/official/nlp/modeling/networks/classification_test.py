# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.networks import classification


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class ClassificationTest(keras_parameterized.TestCase):

  def test_network_creation(self):
    """Validate that the Keras object can be created."""
    input_width = 512
    num_classes = 10
    test_object = classification.Classification(
        input_width=input_width, num_classes=num_classes)
    # Create a 2-dimensional input (the first dimension is implicit).
    cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
    output = test_object(cls_data)

    # Validate that the outputs are of the expected shape.
    expected_output_shape = [None, num_classes]
    self.assertEqual(expected_output_shape, output.shape.as_list())

  def test_network_invocation(self):
    """Validate that the Keras object can be invoked."""
    input_width = 512
    num_classes = 10
    test_object = classification.Classification(
        input_width=input_width, num_classes=num_classes, output='predictions')
    # Create a 2-dimensional input (the first dimension is implicit).
    cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
    output = test_object(cls_data)

    # Invoke the network as part of a Model.
    model = tf.keras.Model(cls_data, output)
    input_data = 10 * np.random.random_sample((3, input_width))
    _ = model.predict(input_data)

  def test_network_invocation_with_internal_logits(self):
    """Validate that the logit outputs are correct."""
    input_width = 512
    num_classes = 10
    test_object = classification.Classification(
        input_width=input_width, num_classes=num_classes, output='predictions')

    # Create a 2-dimensional input (the first dimension is implicit).
    cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
    output = test_object(cls_data)
    model = tf.keras.Model(cls_data, output)
    logits_model = tf.keras.Model(test_object.inputs, test_object.logits)

    batch_size = 3
    input_data = 10 * np.random.random_sample((batch_size, input_width))
    outputs = model.predict(input_data)
    logits = logits_model.predict(input_data)

    # Ensure that the tensor shapes are correct.
    expected_output_shape = (batch_size, num_classes)
    self.assertEqual(expected_output_shape, outputs.shape)
    self.assertEqual(expected_output_shape, logits.shape)

    # Ensure that the logits, when softmaxed, create the outputs.
    input_tensor = tf.keras.Input(expected_output_shape[1:])
    output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
    softmax_model = tf.keras.Model(input_tensor, output_tensor)

    calculated_softmax = softmax_model.predict(logits)
    self.assertAllClose(outputs, calculated_softmax)

  def test_network_invocation_with_internal_and_external_logits(self):
    """Validate that the logit outputs are correct."""
    input_width = 512
    num_classes = 10
    test_object = classification.Classification(
        input_width=input_width, num_classes=num_classes, output='logits')

    # Create a 2-dimensional input (the first dimension is implicit).
    cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
    output = test_object(cls_data)
    model = tf.keras.Model(cls_data, output)
    logits_model = tf.keras.Model(test_object.inputs, test_object.logits)

    batch_size = 3
    input_data = 10 * np.random.random_sample((batch_size, input_width))
    outputs = model.predict(input_data)
    logits = logits_model.predict(input_data)

    # Ensure that the tensor shapes are correct.
    expected_output_shape = (batch_size, num_classes)
    self.assertEqual(expected_output_shape, outputs.shape)
    self.assertEqual(expected_output_shape, logits.shape)

    self.assertAllClose(outputs, logits)

  def test_network_invocation_with_logit_output(self):
    """Validate that the logit outputs are correct."""
    input_width = 512
    num_classes = 10
    test_object = classification.Classification(
        input_width=input_width, num_classes=num_classes, output='predictions')
    logit_object = classification.Classification(
        input_width=input_width, num_classes=num_classes, output='logits')
    logit_object.set_weights(test_object.get_weights())

    # Create a 2-dimensional input (the first dimension is implicit).
    cls_data = tf.keras.Input(shape=(input_width,), dtype=tf.float32)
    output = test_object(cls_data)
    logit_output = logit_object(cls_data)

    model = tf.keras.Model(cls_data, output)
    logits_model = tf.keras.Model(cls_data, logit_output)

    batch_size = 3
    input_data = 10 * np.random.random_sample((batch_size, input_width))
    outputs = model.predict(input_data)
    logits = logits_model.predict(input_data)

    # Ensure that the tensor shapes are correct.
    expected_output_shape = (batch_size, num_classes)
    self.assertEqual(expected_output_shape, outputs.shape)
    self.assertEqual(expected_output_shape, logits.shape)

    # Ensure that the logits, when softmaxed, create the outputs.
    input_tensor = tf.keras.Input(expected_output_shape[1:])
    output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
    softmax_model = tf.keras.Model(input_tensor, output_tensor)

    calculated_softmax = softmax_model.predict(logits)
    self.assertAllClose(outputs, calculated_softmax)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    network = classification.Classification(
        input_width=128,
        num_classes=10,
        initializer='zeros',
        output='predictions')

    # Create another network object from the first object's config.
    new_network = classification.Classification.from_config(
        network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

  def test_unknown_output_type_fails(self):
    with self.assertRaisesRegex(ValueError, 'Unknown `output` value "bad".*'):
      _ = classification.Classification(
          input_width=128, num_classes=10, output='bad')


if __name__ == '__main__':
  tf.test.main()
