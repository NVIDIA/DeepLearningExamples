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
"""Quantized Dense Layer"""

import tensorflow as tf
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn

from ft_tensorflow_quantization.python.layers.tensor_quantizer import QuantDescriptor, FakeQuantizer
from ft_tensorflow_quantization.python.layers.utils import QuantMixin, pop_quant_desc_in_kwargs

__all__ = ["Dense", "QuantDense"]


# TensorFlow use cls.__name__ as default scope name, so keep the name `Dense` for checkpoint restoring
# there is an alias `QuantDense` below
class Dense(tf.layers.Dense, QuantMixin):
  """Quantized version of tf.layers.Dense

  Apply quantized dense to the incoming data, `y = dequant(quant(x)quant(W) + b)`.

  Quantization descriptors are passed in in kwargs. If not presents, `default_quant_desc_input` and
  `default_quant_desc_kernel` are used.

  Args:
    if_quant: A boolean. Whether do quantization. If False, behavior like the original Dense. Default False.
    others: the same as tf.layers.Dense

  Keyword Arguments:
    quant_desc_input: An instance of :func:`QuantDescriptor <quantization.QuantDescriptor>`.
        Quantization descriptor of input.
    quant_desc_wegiht: An instance of :func:`QuantDescriptor <quantization.QuantDescriptor>`.
        Quantization descriptor of kernel.

  Raises:
    ValueError: If unsupported arguments are passed in.
  """
  default_quant_desc_input = QuantDescriptor('input')
  default_quant_desc_kernel = QuantDescriptor('kernel', axis=1)

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               if_quant=False,
               **kwargs):
    
    self.quant_desc_input, self.quant_desc_kernel = pop_quant_desc_in_kwargs(self.__class__, **kwargs)
    self.if_quant = if_quant

    super().__init__(units=units,
                     activation=activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     trainable=trainable,
                     name=name,
                     **kwargs)

  def build(self, input_shape):
    self.kernel_quantizer = FakeQuantizer(self.quant_desc_kernel, 'kernel_quantizer', self.if_quant)
    self.input_quantizer = FakeQuantizer(self.quant_desc_input, 'input_quantizer', self.if_quant)
    self.aftergemm_quantizer = FakeQuantizer(self.quant_desc_input, 'aftergemm_quantizer', self.if_quant)
    super().build(input_shape)

  def call(self, inputs):
    """Forward pass, modified from `tf.layers.Dense.call`"""
    rank = len(inputs.shape)
    kernel = self.kernel_quantizer(self.kernel)
    inputs = self.input_quantizer(inputs)
    
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = math_ops.cast(inputs, self._compute_dtype)
      if K.is_sparse(inputs):
        outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
      else:
        outputs = gen_math_ops.mat_mul(inputs, kernel)
    
    outputs = self.aftergemm_quantizer(outputs)
    
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs


QuantDense = Dense
