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
"""Max collector for calibrations"""

import numpy as np

__all__ = ["MaxCollector"]


class MaxCollector():
  """Collecting min/max values

  Args:
    axis: None or integer. axis which will have its own max for computing scaling factor.
        If None, collect per tensor min/max. Default None
    track_minmax: A boolean. If true, track all min/max it sees in addtion to the returned calib_min/calib_max.
        Default False
  """

  def __init__(self, axis=None, track_minmax=False):
    self._axis = axis
    self._track_minmax = track_minmax

    self._calib_min = None
    self._calib_max = None
    if self._track_minmax:
      self._min_list = []
      self._max_list = []

  def collect(self, x_np):
    """Collect min/max values

    Args:
      x_np: A numpy array to be processed.

    Raises:
      RuntimeError: when the input shape changed
    """

    if self._axis is None:
      reduce_axis = None
    else:
      reduce_axis = []
      axis = self._axis + len(x_np.shape) if self._axis < 0 else self._axis
      for i in range(len(x_np.shape)):
        if i != axis:
          reduce_axis.append(i)
      reduce_axis = tuple(reduce_axis)

    local_min = np.min(x_np, axis=reduce_axis)
    local_max = np.max(x_np, axis=reduce_axis)

    if self._calib_min is None and self._calib_max is None:
      self._calib_min = local_min
      self._calib_max = local_max
    else:
      if local_min.shape != self._calib_min.shape or local_max.shape != self._calib_max.shape:
        raise RuntimeError("quant min/max shape changed!")
      self._calib_min = np.minimum(self._calib_min, local_min)
      self._calib_max = np.maximum(self._calib_max, local_max)

    if self._track_minmax:
      self._min_list.append(local_min)
      self._max_list.append(local_max)

  def reset(self):
    """Reset the collected values"""
    self._calib_min = None
    self._calib_max = None
    if self._track_minmax:
      self._min_list = []
      self._max_list = []

  # pylint:disable=missing-docstring
  @property
  def calib_min(self):
    return self._calib_min

  @property
  def calib_max(self):
    return self._calib_max

  @property
  def min_list(self):
    return self._min_list

  @property
  def max_list(self):
    return self._max_list

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = "MaxCollector("
    s += "axis={_axis}"
    s += " track_minmax={_track_minmax}"
    s += " calib_min={_calib_min}"
    s += " calib_max={_calib_max}"
    if self._track_minmax:
      s += " min_list={_min_list}"
      s += " max_list={_max_list}"
    s += ")"
    return s.format(**self.__dict__)

  # pylint:enable=missing-docstring
