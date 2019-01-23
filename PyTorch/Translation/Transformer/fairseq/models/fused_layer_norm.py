# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# limitations under the License.import math

import math
import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init

import fused_layer_norm_cuda

class FusedLayerNormAffineFunction(torch.autograd.Function):
  def __init__(self, normalized_shape, eps=1e-6):
    self.normalized_shape = normalized_shape
    self.eps = eps

  def forward(self, input, weight, bias):
    input_ = input.contiguous()
    weight_ = weight.contiguous()
    bias_ = bias.contiguous()
    output, mean, invvar = fused_layer_norm_cuda.forward_affine(
        input_, self.normalized_shape, weight_, bias_, self.eps)
    self.save_for_backward(input_, weight_, bias_, mean, invvar)
    return output

  def backward(self, grad_output):
    input_, weight_, bias_, mean, invvar = self.saved_tensors
    grad_input = grad_weight = grad_bias = None
    grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
        grad_output.contiguous(), mean, invvar,
        input_, self.normalized_shape, 
        weight_, bias_, self.eps)
    return grad_input, grad_weight, grad_bias;
    
class FusedLayerNormFunction(torch.autograd.Function):
  def __init__(self, normalized_shape, eps=1e-6):
    self.normalized_shape = normalized_shape
    self.eps = eps

  def forward(self, input):
    input_ = input.contiguous()
    output, mean, invvar = fused_layer_norm_cuda.forward(
        input_, self.normalized_shape, self.eps)
    self.save_for_backward(input_, mean, invvar)
    return output

  def backward(self, grad_output):
    input_, mean, invvar = self.saved_tensors
    grad_input = None
    grad_input = fused_layer_norm_cuda.backward(
        grad_output.contiguous(), mean, invvar,
        input_, self.normalized_shape,
        self.eps)
    return grad_input

def fused_layer_norm_affine(input, normalized_shape, weight, bias, eps=1e-6):
    return FusedLayerNormAffineFunction(normalized_shape,eps)(input, weight, bias)

def fused_layer_norm(input, normalized_shape, eps=1e-6):
    return FusedLayerNormFunction(normalized_shape,eps)(input)

class FusedLayerNorm(torch.nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = nn.LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(FusedLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        if self.elementwise_affine:
          return FusedLayerNormAffineFunction(self.normalized_shape,self.eps)(
              input, self.weight, self.bias)
        else:
          return FusedLayerNormFunction(self.normalized_shape,self.eps)(
              input)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
