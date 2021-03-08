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
"""calibrator class"""

from collections import Counter
import numpy as np
from scipy.stats import entropy
import tensorflow as tf

from ft_tensorflow_quantization.python.calib.max import MaxCollector
from ft_tensorflow_quantization.python.calib.histogram import HistogramCollector
from ft_tensorflow_quantization.python.utils.utils import scaled_quant_np

__all__ = ["Calibrator", "get_calibrators"]


class Calibrator():
  """A calibrator that wraps up a collector and relavent tensors and does calibration

    Args:
      tensor_name_prefix: A string. The common name prefix of `quant_min`, `quant_max`, and `calib_tensor`.
      collector: :func:`MaxCollector <quantization.MaxCollector>`
          or :func:`HistogramCollector <quantization.HistogramCollector>`.
      quant_min_name: The name of corresponding `quant_min` tensor in the graph.
      quant_max_name:The name of corresponding `quant_max` tensor in the graph.
      calib_tensor_name: The name of the tensor need be calibrated.

    Attributes:
      - tensor_name_prefix: Read-only property for the common name prefix of
          `quant_min`, `quant_max`, and `calib_tensor`.
      - calib_min: Read-only property for the min value the calibrator collected/computed.
      - calib_max: Read-only property for the max value the calibrator collected/computed.
      - quant_min_name: Read-only property for the name of `quant_min` tensor in the fakequant node in the graph.
      - quant_max_name: Read-only property for the name of `quant_max` tensor in the fakequant node in the graph.
    """

  def __init__(self, tensor_name_prefix, collector, quant_min_name, quant_max_name, calib_tensor_name):
    self._tensor_name_prefix = tensor_name_prefix
    self._collector = collector
    self._quant_min_name = quant_min_name
    self._quant_max_name = quant_max_name
    self._calib_tensor_name = calib_tensor_name

    self._calib_min = None
    self._calib_max = None

  def calib_step_op(self, graph):
    """get the op for one step of calibration

    Args:
      graph: The being excuted TensorFlow Graph.

    Returns:
      A wrapped TensorFlow op of `tf.py_function` for one calib step.
    """
    return tf.py_function(self._collector.collect, inp=[graph.get_tensor_by_name(self._calib_tensor_name)], Tout=[])

  def compute_range(self, calibration_method, **kwargs):
    """calculate min/max values from collector
      if :func:`MaxCollector <quantization.MaxCollector>` is used, kwargs should be None.
      if :func:`HistogramCollector <quantization.HistogramCollector>` is used,
      there should be `calibration_method` in kwargs and other corresponding arguments.

    Args:
      calibration_method: A string indicates the calibration method.
          One of `["max", "percentile", "mse", "entropy"]`.

    Keyword Arguments:
      percentile: A float. Set range to p-th percentile of collected data. `0 <= p <= 100`.
          Only needed when `calibration_method == "percentile"`.
      start_bin: An integer. Histogram bin to start sweep. Default 128.
          Only needed when `calibration_method == "mse"` or `calibration_method == "entropy"`.
      stride: An integer. Stride of histogram bins swept. Default 1.
          Only needed when `calibration_method == "mse"` or `calibration_method == "entropy"`.
      num_bits: An integer. Number of bits of quantization. Default 8.
          Only needed when `calibration_method == "mse"` or `calibration_method == "entropy"`.
      unsigned: A boolean. using unsigned quantization. Default False.
          Only needed when `calibration_method == "mse"` or `calibration_method == "entropy"`.

    Raises:
      ValueError: Wrong arguments is provided.
      RuntimeError: compute range before collecting.
    """
    if calibration_method not in ["max", "percentile", "mse", "entropy"]:
      raise ValueError('calibration_method should be one of ["max", "percentile", "mse", "entropy"]')
  
    if isinstance(self._collector, MaxCollector):
      assert calibration_method == "max"
      if self._collector.calib_min is None or self._collector.calib_max is None:
        raise RuntimeError("The collector have not collected anything, cannot compute the range.")
      if kwargs:
        raise ValueError("Unexpected keys: {}".format(kwargs.keys()))
      self._calib_min, self._calib_max = self._collector.calib_min, self._collector.calib_max

    elif isinstance(self._collector, HistogramCollector):
      if self._collector.calib_bin_edges is None or self._collector.calib_hist is None:
        raise RuntimeError("The collector have not collected anything, cannot compute the range.")

      if calibration_method == 'percentile':
        percentile = kwargs.pop('percentile', None)
        if percentile is None:
          raise ValueError("A percentile value should be provided")
        if kwargs:
          raise ValueError("Unexpected keys: {}".format(kwargs.keys()))
        self._calib_min, self._calib_max = self._compute_percentile_range(percentile)

      elif calibration_method in ['mse', 'entropy']:
        start_bin = kwargs.pop('start_bin', 128)
        stride = kwargs.pop('stride', 1)
        num_bits = kwargs.pop('num_bits', 8)
        unsigned = kwargs.pop('unsigned', False)
        if kwargs:
          raise ValueError("Unexpected keys: {}".format(kwargs.keys()))
        if calibration_method == 'mse':
          self._calib_min, self._calib_max = self._compute_mse_range(start_bin, stride, num_bits, unsigned)
        else:
          self._calib_min, self._calib_max = self._compute_entropy_range(start_bin, stride, num_bits, unsigned)

      else:
        raise ValueError("calibration_method must be one of ['percentile', 'mse', 'entropy']")

  def _compute_percentile_range(self, percentile):
    """compute min/max value with percentile method and return a tuple of (min, max)
      Choose min/max to clip the top P percentile of data
    """
    if percentile < 0 or percentile > 100:
      raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

    if not self._collector.affine:
      total = self._collector.calib_hist.sum()
      cdf = np.cumsum(self._collector.calib_hist / total)
      idx = np.searchsorted(cdf, percentile / 100)
      calib_max = self._collector.calib_bin_edges[idx]
      result = -calib_max.astype('float32'), calib_max.astype('float32')
    else:
      raise NotImplementedError("No affine support for now.")
    return result

  def _compute_mse_range(self, start_bin, stride, num_bits, unsigned):
    """compute min/max value that minimizes MSE of the collected histogram
        and return a tuple of (min, max)
    """

    if not self._collector.affine:
      centers = (self._collector.calib_bin_edges[1:] + self._collector.calib_bin_edges[:-1]) / 2
      mses = []
      arguments = []
      for i in range(start_bin, len(centers), stride):
        amax = centers[i]
        quant_centers = scaled_quant_np(centers, amax, num_bits, axis=None, unsigned=unsigned)
        mse = ((quant_centers - centers)**2 * self._collector.calib_hist).mean()
        mses.append(mse)
        arguments.append(i)
      argmin = np.argmin(mses)
      calib_max = centers[arguments[argmin]]
      result = -calib_max.astype('float32'), calib_max.astype('float32')
    else:
      raise NotImplementedError("No affine support for now.")
    return result

  def _compute_entropy_range(self, start_bin, stride, num_bits, unsigned):
    """compute min/max value that minimizes KL-Divergence of the collected histogram
        and return a tuple of (min, max)
    """

    def _normalize_distr(distr):
      summ = np.sum(distr)
      if summ != 0:
        distr = distr / summ

    if not self._collector.affine:
      bins = self._collector.calib_hist[:]
      bins[0] = bins[1]

      total_data = np.sum(bins)

      divergences = []
      arguments = []

      # we are quantizing to 128 values + sign if num_bits=8
      nbins = 1 << (num_bits - 1 + int(unsigned))
      stop = len(bins)

      new_density_counts = np.zeros(nbins, dtype=np.float64)

      for i in range(start_bin, stop + 1, stride):
        new_density_counts.fill(0)
        space = np.linspace(0, i, num=nbins + 1)
        digitized_space = np.digitize(range(i), space) - 1

        digitized_space[bins[:i] == 0] = -1

        for idx, digitized in enumerate(digitized_space):
          if digitized != -1:
            new_density_counts[digitized] += bins[idx]

        counter = Counter(digitized_space)
        for key, val in counter.items():
          if key != -1:
            new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        for idx, digitized in enumerate(digitized_space):
          if digitized != -1:
            new_density[idx] = new_density_counts[digitized]

        total_counts_new = np.sum(new_density) + np.sum(bins[i:])
        _normalize_distr(new_density)

        reference_density = np.array(bins[:len(digitized_space)])
        reference_density[-1] += np.sum(bins[i:])

        total_counts_old = np.sum(reference_density)
        if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
          raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
              total_counts_new, total_counts_old, total_data))

        _normalize_distr(reference_density)

        ent = entropy(reference_density, new_density)
        divergences.append(ent)
        arguments.append(i)

      divergences = np.array(divergences)
      last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
      calib_max = self._collector.calib_bin_edges[last_argmin * stride + start_bin]

      result = -calib_max.astype('float32'), calib_max.astype('float32')
    else:
      raise NotImplementedError("No affine support for now.")
    return result

  def load_range(self, sess):
    """load min/max values to the graph

    Args:
      sess: A TensorFlow Session.
    """
    if self._calib_min is None or self._calib_max is None:
      raise RuntimeError("load_range should be called after compute_range")
    sess.run(tf.compat.v1.assign(sess.graph.get_tensor_by_name(self._quant_min_name), tf.constant(self._calib_min)))
    sess.run(tf.compat.v1.assign(sess.graph.get_tensor_by_name(self._quant_max_name), tf.constant(self._calib_max)))

  def compute_and_load_range(self, sess, **compute_range_args):
    """wraps :func:`compute_range <quantization.Calibrator.compute_range>`
    and :func:`load_range <quantization.Calibrator.load_range>` for convinience"""
    self.compute_range(**compute_range_args)
    self.load_range(sess)

  # pylint:disable=missing-docstring
  @property
  def tensor_name_prefix(self):
    return self._tensor_name_prefix

  @property
  def calib_min(self):
    if self._calib_min is None:
      raise RuntimeError("Accessing calib_min need compute_range called first.")
    return self._calib_min

  @property
  def calib_max(self):
    if self._calib_max is None:
      raise RuntimeError("Accessing calib_max need compute_range called first.")
    return self._calib_max

  @property
  def quant_min_name(self):
    return self._quant_min_name

  @property
  def quant_max_name(self):
    return self._quant_max_name

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return "Calibrator({})".format(self._tensor_name_prefix)

  # pylint:enable=missing-docstring


def get_calibrators(collection_name_prefix,
                    graph=None,
                    collector_type='max',
                    **collector_args):
  """Prepare collector and relevant tensors for calibration and return a list of calibrators.

  Args:
    collection_name_prefix: A string. Determine the collection of tensors. Need to be unified with FakeQuantizer.
    graph: an instance of `tf.Graph`, if None, use default graph. Default None.
    collector_types: A string. What collector to use. One of `["max", "histogram"]`. Default `"max"`.
    Collector arugments can be passed by collector_args.
        If :func:`MaxCollector <quantization.MaxCollector>` is used,
        only `axis` and `track_minmax` can be passed to collector_args.
        If :func:`HistogramCollector <quantization.HistogramCollector>` is used,
        only `num_bins`, `grow_method`, `skip_zeros` and `affine` can be passed.
        For details of these arguments, please refer to the docs of :func:`MaxCollector <quantization.MaxCollector>`
        or :func:`HistogramCollector <quantization.HistogramCollector>`.

  Return:
    A list of calibrators. Each calibrator processes tensors
        in a corresponding :func:`FakeQuantizer <quantization.FakeQuantizer>`.
  """
  if graph is None:
    graph = tf.compat.v1.get_default_graph()

  qmin_collection = graph.get_collection(collection_name_prefix + '_quant_min')
  qmax_collection = graph.get_collection(collection_name_prefix + '_quant_max')
  calib_tensor_collection = graph.get_collection(collection_name_prefix + '_calib_tensor')

  collection_size = len(calib_tensor_collection)
  assert len(qmin_collection) == collection_size
  assert len(qmax_collection) == collection_size

  def get_name_prefix(tensor_name):
    tensor_name = tensor_name.split('/')
    prefix = '/'.join(tensor_name[:-1])
    return prefix

  def verify_collector_args(collector_args, acceptable_args, collector_name):
    for k, _ in collector_args.items():
      if k not in acceptable_args:
        raise ValueError("Wrong arguments {} for {} collector, only {} are supported.".format(
            k, collector_name, acceptable_args))

  if collector_type == 'max':
    verify_collector_args(collector_args, ['axis', 'track_minmax'], collector_type)
    collector_class = MaxCollector
  elif collector_type == 'histogram':
    verify_collector_args(collector_args, ['num_bins', 'grow_method', 'skip_zeros', 'affine'], collector_type)
    collector_class = HistogramCollector
  else:
    raise ValueError("collector_type must be one of ['max', 'histogram']")

  result = []
  for i in range(collection_size):
    name_prefix = get_name_prefix(calib_tensor_collection[i])
    assert get_name_prefix(qmin_collection[i]) == name_prefix
    assert get_name_prefix(qmax_collection[i]) == name_prefix

    calibrator = Calibrator(name_prefix, collector_class(**collector_args), qmin_collection[i], qmax_collection[i],
                            calib_tensor_collection[i])
    result.append(calibrator)

  return result
