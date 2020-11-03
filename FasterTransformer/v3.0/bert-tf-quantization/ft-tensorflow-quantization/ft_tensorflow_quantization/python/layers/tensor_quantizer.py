################################################################################
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
################################################################################
"""Tensor quantizer"""

import tensorflow as tf

from ft_tensorflow_quantization.python.ops.fake_quantize import fake_quantize

__all__ = ["QuantDescriptor", "FakeQuantizer"]


class QuantDescriptor():
  """Supportive descriptor of quantization. Describe how a tensor should be quantized.

  Args:
    collection_name_prefix: A string. Determine which collection to put for reverant tensors.
    num_bits: An integer. Number of bits of quantization. It is used to calculate scaling factor. Default 8.

  Keyword Arguments:
    scope_name: A string. The name is used to define the name scope in the quantizer. Better to special.
        Default `"tensor_quantizer"`.
    axis: None or integer. axes which will have its own max for computing scaling factor.
        If None (the default), use per tensor scale.
        Must be in the range `[-rank(input_tensor), rank(input_tensor)]`.
        e.g. For a KCRS weight tensor, `axis=0` will yield per channel scaling.
        Default None.
    unsigned: A Boolean. If True, use unsigned. Default False.
    affine: A Boolean. If True, use affine quantization. Default False.

  Raises:
    TypeError: Wrong argument type.
    ValueError:Wrong argument value.

  Attributes:
    - collection_name_prefix: read-only property.
    - scope_name: read-only property.
    - num_bits: read-only property.
    - unsigned: read-only property.
    - affine: read-only property.
    - axis: read-only property.
  """

  def __init__(self, collection_name_prefix, num_bits=8, **kwargs):
    if not isinstance(num_bits, int):
      raise TypeError("num_bits must be an integer, not {}.".format(type(num_bits)))
    if num_bits <= 0:
      raise ValueError("num_bits must be > 0, not {}.".format(num_bits))
    self._num_bits = num_bits

    self._unsigned = kwargs.pop('unsigned', False)
    self._affine = kwargs.pop('affine', False)
    self._axis = kwargs.pop('axis', None)

    self._collection_name_prefix = collection_name_prefix
    if not isinstance(self._collection_name_prefix, str):
      raise TypeError("collection_name_prefix must be a string, not {}.".format(type(self._collection_name_prefix)))
    self._scope_name = kwargs.pop('scope_name', "tensor_quantizer")
    if not isinstance(self._scope_name, str):
      raise TypeError("scope_name must be a string, not {}.".format(type(self._scope_name)))

    if kwargs:
      raise TypeError("Unrecognized keys: {}".format(kwargs.keys()))

  # pylint:disable=missing-docstring
  @property
  def collection_name_prefix(self):
    return self._collection_name_prefix

  @property
  def scope_name(self):
    return self._scope_name

  @property
  def num_bits(self):
    return self._num_bits

  @property
  def unsigned(self):
    return self._unsigned

  @property
  def affine(self):
    return self._affine

  @property
  def axis(self):
    return self._axis

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = "QuantDescriptor("
    s += "num_bits={_num_bits}"
    s += " unsigned={_unsigned}"
    s += " affine={_affine}"
    s += " axis={_axis}"
    s += " collection_name_prefix='{_collection_name_prefix}'"
    s += " scope_name='{_scope_name}'"
    s += ")"
    return s.format(**self.__dict__)

  # pylint:enable=missing-docstring


class FakeQuantizer():
  """Fake Tensor quantizer module

  This module quantize a tensor and wraps variable. It also can collect relevant variables for calibration.

  Args:
    quant_desc: An instance of :func:`QuantDescriptor <quantization.QuantDescriptor>`.
    if_quant: A boolean. Determine whether do quantization or not. Default True.
        If False, quantizaton will be disabled.
        This quantizer will always set collections for calibration.

  Raises:
    TypeError: when wrong type of `quant_desc`.
  """

  def __init__(self, quant_desc: QuantDescriptor, if_quant=True):
    if not isinstance(quant_desc, QuantDescriptor):
      raise TypeError("quant_desc should be a QuantDescriptor")

    self._num_bits = quant_desc.num_bits
    self._axis = quant_desc.axis
    self._unsigned = quant_desc.unsigned
    self._affine = quant_desc.affine
    self._collection_name_prefix = quant_desc.collection_name_prefix
    self._scope_name = quant_desc.scope_name

    self._if_quant = if_quant

    self._quant_min = None
    self._quant_max = None

  # pylint:disable=missing-docstring
  @property
  def quant_min(self):
    return self._quant_min

  @property
  def quant_max(self):
    return self._quant_max

  # pylint:enable=missing-docstring

  def __call__(self, inputs):

    if self._axis is None:
      quant_shape = tuple()
    else:
      quant_shape = (inputs.shape.as_list()[self._axis],)

    with tf.compat.v1.variable_scope(None, default_name=self._scope_name):
      self._quant_min = tf.compat.v1.get_variable("quant_min", shape=quant_shape, trainable=False)
      self._quant_max = tf.compat.v1.get_variable("quant_max", shape=quant_shape, trainable=False)
      # add tensor to collection `quantization_variables` to convinient initializing from checkpoint
      tf.compat.v1.add_to_collection('quantization_variables', self._quant_min)
      tf.compat.v1.add_to_collection('quantization_variables', self._quant_max)
      # add tensor name to collections for calibration
      tf.compat.v1.add_to_collection(self._collection_name_prefix + '_quant_min', self._quant_min.name)
      tf.compat.v1.add_to_collection(self._collection_name_prefix + '_quant_max', self._quant_max.name)
      # use identity to put these variables to a unified name scope for calibration
      tensor_for_calib = tf.identity(inputs, name="tensor_for_calib")
      tf.compat.v1.add_to_collection(self._collection_name_prefix + '_calib_tensor', tensor_for_calib.name)

      if self._if_quant:
        outputs = fake_quantize(inputs, self._quant_min, self._quant_max, self._num_bits, self._axis, self._unsigned,
                                self._affine)
      else:
        outputs = inputs

    return outputs
