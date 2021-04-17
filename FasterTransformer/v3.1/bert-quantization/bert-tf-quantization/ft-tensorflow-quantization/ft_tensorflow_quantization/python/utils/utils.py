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
"""utilities"""

import numpy as np


def expand_dims(x, v, axis):
  if axis < 0:
    axis += len(x.shape)
  for i in range(len(x.shape)):
    if i != axis:
      v = np.expand_dims(v, i)
  return v


def scaled_quant_np(x, amax, num_bits=8, axis=None, unsigned=False):
  """Scaled quantize x using numpy."""
  if axis is not None:
    amax = expand_dims(x, amax, axis)

  quant_bound = 2.0**(num_bits - 1 + int(unsigned)) - 1
  quant_scale = quant_bound / amax
  x_q = np.round(np.clip(x, -amax, amax) * quant_scale)
  x_q /= quant_scale
  return x_q


def affine_quant_np(x, qmin, qmax, num_bits=8, axis=None, unsigned=False):
  """Affine quantize x using numpy."""
  if axis is not None:
    qmin = expand_dims(x, qmin, axis)
    qmax = expand_dims(x, qmax, axis)

  if unsigned:
    min_bound = 0
    max_bound = 2.0**num_bits - 1.0
  else:
    min_bound = -2.0**(num_bits - 1)
    max_bound = 2.0**(num_bits - 1) - 1.0

  step_size = (qmax - qmin) / (2.0**num_bits - 1.0)
  quant_zero = np.round(qmin / step_size) - min_bound
  x_q = np.round(x / step_size) - quant_zero
  x_q = np.clip(x_q, min_bound, max_bound)
  x_q = (x_q + quant_zero) * step_size
  return x_q
