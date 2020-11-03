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
"""Basic tensor quantization functions"""

import tensorflow as tf

__all__ = ["fake_quantize"]


def fake_quantize(inputs, quant_min=None, quant_max=None, num_bits=8, axis=None, unsigned=False, affine=False):
  """Universal tensor fake quantization function

  Args:
    inputs: A Tensor of dtype float32.
    quant_min: Scalar (0-d Tensor), 1-d Tensor or None
    quant_max: Scalar (0-d Tensor), 1-d Tensor or None.
    num_bits: An integer used to calculate scaling factor, `scale = (2^(num_bits-1) - 1) / max`.
        Effectively, it indicates how many integer bits is used to represent the value.
    axis: Integer or None. If specified, quant_min and quant_max must be vector and will be broadcasted to inputs.
        Default None, indicates per tensor quantization.
    unsigned: A boolean. If True, use unsigned int8. Default False.
    affine: A boolean. If True, use affine quantization. Default False.

  Returns:
    outputs: A Tensor with same type as inputs

  Raises:
    TypeError: Wrong input types.
    ValueError: Wrong input values.
  """
  if not tf.is_tensor(inputs):
    raise TypeError("inputs should be a Tensor")
  if not isinstance(num_bits, int):
    raise TypeError("num_bits should be an integer")
  if num_bits <= 0:
    raise ValueError("num_bits should > 0")

  if quant_max is None and quant_min is None:
    raise NotImplementedError("dynamic quantization is not supported yet")
  if quant_min is not None and quant_max is not None:
    if not tf.is_tensor(quant_max) or not tf.is_tensor(quant_min):
      raise TypeError("quant_min and quant_max should be Scalar (0-d Tensor), 1-d Tensor or None")
    if quant_max.shape != quant_min.shape:
      raise ValueError("shape mismatch between quant_min and quant_max")
    if len(quant_max.shape) == 0:
      if axis is not None:
        raise ValueError("quan_min/quant_max is a Scalar, support per tensor quantization, axis must be None")
    elif len(quant_max.shape) == 1:
      if axis is None:
        raise ValueError("quan_min/quant_max is a Tensor, support per axis quantization, axis must be set")
      if not isinstance(axis, int):
        raise TypeError("axis should be an integer")
      if not -len(inputs.shape) <= axis < len(inputs.shape):
        raise ValueError("invalid axis {} for inputs with dimentsion {}".format(axis, len(inputs.shape)))
    else:
      raise ValueError("quant_min and quant_max should be Scalar (0-d Tensor), 1-d Tensor or None")
  else:
    raise ValueError("one of quant_min and quant_max is None")

  # do broadcast obviously for per axis quantization
  if axis is not None:
    if axis < 0:
      axis += len(inputs.shape)
    for i in range(len(inputs.shape)):
      if i != axis:
        quant_min = tf.expand_dims(quant_min, i)
        quant_max = tf.expand_dims(quant_max, i)

  epsilon = 1. / (1 << 24)  # Minimum fp16 representable

  @tf.custom_gradient
  def fake_quantize_core(inputs, quant_min, quant_max):

    def _scaled_fake_quantize(inputs, quant_min, quant_max):
      # TODO(haow): Add check for negative values in inputs if unsigned
      bound = 2.0**(num_bits - 1 + int(unsigned)) - 1.0
      quant_amax = tf.maximum(tf.abs(quant_min), tf.abs(quant_max))
      scale = bound / quant_amax

      # Treat quant_max smaller than minimum representable of fp16 0.
      # Value quantized with quant_amax=0 should all be 0, thus set scale to 1
      scale = tf.compat.v2.where(tf.math.less_equal(quant_amax, epsilon), tf.constant(1.), scale)

      quantized = tf.clip_by_value(tf.math.round(inputs * scale), -bound, bound)
      outputs = quantized / scale
      return outputs

    def _affine_fake_quantize(inputs, quant_min, quant_max):
      if unsigned:
        min_bound = 0
        max_bound = 2.0**num_bits - 1.0
        zero_point = 2.0**(num_bits - 1)
      else:
        min_bound = -2.0**(num_bits - 1)
        max_bound = 2.0**(num_bits - 1) - 1.0
        zero_point = 0.0
      step_size = (quant_max - quant_min) / (2.0**num_bits - 1.0)
      # in case step_size is too small, may need improved
      quant_zero = tf.compat.v2.where(tf.math.less_equal(step_size, epsilon), tf.constant(zero_point),
                                      tf.math.round(quant_min / step_size) - min_bound)
      quantized = tf.compat.v2.where(
          tf.math.less_equal(step_size, epsilon), quant_zero,
          tf.clip_by_value(tf.math.round(inputs / step_size) - quant_zero, min_bound, max_bound))
      outputs = tf.compat.v2.where(tf.math.less_equal(step_size, epsilon), quant_max,
                                   (quantized + quant_zero) * step_size)
      return outputs

    if not affine:
      outputs = _scaled_fake_quantize(inputs, quant_min, quant_max)
    else:
      outputs = _affine_fake_quantize(inputs, quant_min, quant_max)

    def grad(grad_outputs):
      # Boundary check is exclusive in case we'll need to support inplace
      if not affine:
        pass_condition = tf.math.less(tf.abs(inputs), tf.maximum(tf.abs(quant_min), tf.abs(quant_max)))
      else:
        pass_condition = tf.math.logical_and(tf.math.greater(inputs, quant_min), tf.math.less(inputs, quant_max))
      grad_inputs = tf.compat.v2.where(pass_condition, grad_outputs, tf.constant(0.))
      return grad_inputs, None, None

    return outputs, grad

  return fake_quantize_core(inputs, quant_min, quant_max)
