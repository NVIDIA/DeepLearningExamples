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
"""Histogram collector"""

import numpy as np

__all__ = ["HistogramCollector"]


class HistogramCollector():
  """Collecting histograms and do calibration

  Args:
    num_bins: An integer. Number of histograms bins. Default 2048
    grow_method: A string. Method to grow histogram, `append` or `stretch`. Default `append`.
        If 'stretch', increase the size of the last bin to capture outliers.
        If 'append', add more bins of the same size.
    skip_zeros: A boolean. count zeros in data. Default False
    affine: A boolean. If True, collect histogram for affine quantization. Default False.

  Raises:
    ValueError: If invalid grow_method is given.
  """

  def __init__(self, num_bins=2048, grow_method='append', skip_zeros=False, affine=False):
    self._num_bins = num_bins
    if grow_method not in ['stretch', 'append']:
      raise ValueError("grow_method must be one of 'stretch', 'append'")
    self._grow_method = grow_method
    self._skip_zeros = skip_zeros
    self._affine = affine

    self._calib_bin_edges = None
    self._calib_hist = None

  def collect(self, x_np):
    """Collect histogram

    Args:
      x_np: A numpy array to be processed.
    """

    if self._skip_zeros:
      x_np = x_np[np.where(x_np != 0)]

    if not self._affine:
      x_np = np.abs(x_np)
    else:
      raise NotImplementedError("No affine support for now.")

    temp_max = np.max(x_np)
    if self._calib_bin_edges is None and self._calib_hist is None:
      # first time it uses num_bins to compute histogram.
      width = temp_max / self._num_bins
      self._calib_bin_edges = np.arange(0, temp_max + width, width)
      self._calib_hist, self._calib_bin_edges = np.histogram(x_np, bins=self._calib_bin_edges)
    else:
      width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
      if temp_max > self._calib_bin_edges[-1]:
        if self._grow_method == 'append':
          # increase the number of bins
          self._calib_bin_edges = np.arange(self._calib_bin_edges[0], temp_max + width, width)
        elif self._grow_method == 'stretch':
          # stretch the last bin edge to capture the new range
          self._calib_bin_edges[-1] = temp_max
        else:
          raise ValueError("unknown grow_method '{}'".format(self._grow_method))

      hist, self._calib_bin_edges = np.histogram(x_np, bins=self._calib_bin_edges)
      hist[:len(self._calib_hist)] += self._calib_hist
      self._calib_hist = hist

  def reset(self):
    """Reset the collected histogram"""
    self._calib_bin_edges = None
    self._calib_hist = None

  # pylint:disable=missing-docstring
  @property
  def calib_bin_edges(self):
    return self._calib_bin_edges

  @property
  def calib_hist(self):
    return self._calib_hist

  @property
  def affine(self):
    return self._affine

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = "HistogramCollector("
    s += "num_bins={_num_bins}"
    s += " grow_method={_grow_method}"
    s += " skip_zeros={_skip_zeros}"
    s += " affine={_affine}"
    s += " calib_bin_edges={_calib_bin_edges}"
    s += " calib_hist={_calib_hist})"
    return s.format(**self.__dict__)

  # pylint:enable=missing-docstring
