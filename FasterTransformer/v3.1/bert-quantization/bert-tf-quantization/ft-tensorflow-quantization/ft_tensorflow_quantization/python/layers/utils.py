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
"""Some helper functions for implementing quantized layers"""

import copy

from ft_tensorflow_quantization.python.layers.tensor_quantizer import QuantDescriptor


class QuantMixin():
  """Mixin class for adding basic quantization logic to quantized modules"""

  default_quant_desc_input = QuantDescriptor('input')
  default_quant_desc_kernel = QuantDescriptor('kernel', axis=-1)

  @classmethod
  def set_default_quant_desc_input(cls, value):
    """
    Args:
        value: An instance of :func:`QuantDescriptor <quantization.QuantDescriptor>`
    """
    if not isinstance(value, QuantDescriptor):
      raise ValueError("{} is not an instance of QuantDescriptor!")
    cls.default_quant_desc_input = copy.deepcopy(value)

  @classmethod
  def set_default_quant_desc_kernel(cls, value):
    """
    Args:
        value: An instance of :func:`QuantDescriptor <quantization.QuantDescriptor>`
    """
    if not isinstance(value, QuantDescriptor):
      raise ValueError("{} is not an instance of QuantDescriptor!")
    cls.default_quant_desc_kernel = copy.deepcopy(value)


def pop_quant_desc_in_kwargs(quant_cls, **kwargs):
  """Pop quant descriptors in kwargs

  If there is no descriptor in kwargs, the default one in quant_cls will be used

  Arguments:
    quant_cls: A class that has default quantization descriptors

  Keyword Arguments:
    quant_desc_input: An instance of QuantDescriptor. Quantization descriptor of input.
    quant_desc_kernel: An instance of QuantDescriptor. Quantization descriptor of kernel.
  """
  quant_desc_input = kwargs.pop('quant_desc_input', quant_cls.default_quant_desc_input)
  quant_desc_kernel = kwargs.pop('quant_desc_kernel', quant_cls.default_quant_desc_kernel)
  # base layers may use kwargs, so do not check if anything is left in **kwargs
  return quant_desc_input, quant_desc_kernel
