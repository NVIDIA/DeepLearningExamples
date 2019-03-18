# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Contains classes and functions for doing a single-machine batch all-reduce.

An all-reduce is taking the reduction (typically a sum) of a list of tensors,
each on a different device. The result must end up back on each device, which is
where the word "all" comes from. In summary, each device starts with a single
tensor, and ends up with the reduction of all tensors.

A batch all-reduce is doing several independent all-reduces. When doing a batch
all-reduce, care is taken to evenly distribute the reduction computations
across devices and inter-device tensor transfers across device links.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(reedwm): Support distributed all-reduces in this file.
# TODO(reedwm): Merge this code with allreduce.py, which contains some batch
# all-reduce code that this file calls. allreduce.py also supports distributed
# batch-reduce while this file only supports single-machine all-reduce.

import abc
from collections import namedtuple

import six
import tensorflow as tf

from tensorflow.python.ops import gradients_impl

from variable_mgr import allreduce
from variable_mgr import constants


def _all_reduce_using_copy(tensors_across_devices, use_mean):
  """Does an all-reduce of a list of tensors by copying to the current device.

  The tensors are copied to the current device and then reduced.

  Args:
    tensors_across_devices: A list of tensors, each on a different device.
    use_mean: Whether to take the mean of the tensors instead of a sum:
  Returns:
    A reduced tensor on the current device.
  """
  assert tensors_across_devices
  if isinstance(tensors_across_devices[0], tf.IndexedSlices):
    reduced_tensor = gradients_impl._AggregateIndexedSlicesGradients(
        tensors_across_devices)
    if use_mean:
      val = tf.multiply(reduced_tensor.values,
                        float(1. / len(tensors_across_devices)))
      reduced_tensor = tf.IndexedSlices(val, reduced_tensor.indices,
                                        reduced_tensor.dense_shape)
  else:
    reduced_tensor = tf.add_n(tensors_across_devices)
    if use_mean:
      reduced_tensor *= 1. / len(tensors_across_devices)
  return reduced_tensor


@six.add_metaclass(abc.ABCMeta)
class BatchAllReduceAlgorithm(object):
  """Represents an algorithm for performing a batch all-reduce operation."""

  def batch_all_reduce(self, all_device_tensors, num_splits, compact_tensors,
                       defer_tensors):
    """Performs a batch all-reduce.

    The reduction done is a sum.

    `all_device_tensors` is a list of list of tensors that will be batch
    all-reduced. All tensors within a single inner list must be on the same
    device. The nth element in each list, for any n, will be reduced together.
    The return value is in the same form as `all_device_tensors`, except that
    each tensor is reduced.

    For example, if `all_device_tensors` is:
    [[ A,  B  ],     # A and B are on GPU 0
     [ C,  D  ]]     # C and D are on GPU 1

    Then the return value will be:
    [[ A+C,  B+D ],  # These two tensors are on GPU 0
     [ A+C,  B+D ]]  # These two tensors are on GPU 1

    Arguments:
      all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]`
        is a tensor where `i` is the device index and `j` is the tensor index.
      num_splits: If not None, tensors will be concatenated and split into this
        many pieces during the all-reduce, then split back into their original
        shapes afterwards. Has no impact on correctness and can improve
        performance. Requires all tensors to be the same type.
      compact_tensors: If True, tensors are casted to fp16 before being all-
        reduced. Improves performance, but hurts numerical stability.
      defer_tensors: If True, every time the return value
        `reduced_all_device_tensors` is evaluated, the result will be the
        reduced tensors values of `all_device_tensors` from the previous session
        run instead of the current session run, or zero on the first session
        run. This can improve performance. When training neural networks,
        deferring gradients often does not harm training, so this can be used to
        improve performance.

    Returns:
      reduced_all_device_tensors: A list in the same form as
        `all_device_tensors`, except each tensor has been reduced.
      warmup_ops: A list of ops needed to be run once before the all-reduce can
        occur.
    """
    # Before all-reducing tensors, we do several preprocessing functions that
    # can speed up the all-reduce. We undo these functions after all-reducing
    # the tensors.
    warmup_ops = []
    if num_splits:
      packer = _TensorPacker(num_splits)
      all_device_tensors = packer.concat_all_device_tensors(all_device_tensors)
    # If enabled, we compact and defer tensors in between concatenating them
    # and splitting them, because it is faster to do operations on a single
    # concatenated tensor than on multiple smaller tensors.
    if compact_tensors:
      all_device_tensors_before_compact = all_device_tensors
      all_device_tensors = _compact_all_device_tensors(all_device_tensors)
    if defer_tensors:
      all_device_tensors, put_ops, warmup_ops = _defer_all_device_tensors(
          all_device_tensors)
    if num_splits:
      all_device_tensors = packer.split_all_device_tensors(all_device_tensors)

    all_device_tensors = self._do_batch_all_reduce(all_device_tensors)

    # Undo the preprocessing operations in opposite order as we applied them.
    if num_splits:
      all_device_tensors = packer.undo_split_all_device_tensors(
          all_device_tensors)
    # Note: There is no undo operation for deferring tensors. But we do need to
    # call _add_put_op_control_deps at the end if we deferred the tensors.
    if compact_tensors:
      all_device_tensors = _undo_compact_all_device_tensors(
          all_device_tensors, all_device_tensors_before_compact)
    if num_splits:
      all_device_tensors = packer.undo_concat_all_device_tensors(
          all_device_tensors)

    if defer_tensors:
      all_device_tensors = _add_put_op_control_deps(all_device_tensors,
                                                    num_splits, put_ops)
    return all_device_tensors, warmup_ops

  @abc.abstractmethod
  def _do_batch_all_reduce(self, all_device_tensors):
    """Performs a batch all-reduce.

    Unlike `self.batch_all_reduce`, this does not do any preprocessing of the
    tensors.

    Args:
      all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]`
        is a tensor where `i` is the device index and `j` is the tensor index.
    Returns:
      reduced_all_device_tensors: A list in the same form as
        `all_device_tensors`, except each tensor has been reduced.
    """
    pass


class CopyToDeviceAlgorithm(BatchAllReduceAlgorithm):
  """An algorithm that copies tensors to be reduced to a specific device."""

  def __init__(self, devices_to_reduce_on, use_mean=False):
    self._devices = devices_to_reduce_on
    self._use_mean = use_mean

  def _do_batch_all_reduce(self, all_device_tensors):
    reduced_tensors = []
    for i, tensors_across_devices in enumerate(zip(*all_device_tensors)):
      with tf.device(self._devices[i % len(self._devices)]):
        reduced_tensor = _all_reduce_using_copy(tensors_across_devices,
                                                self._use_mean)
        reduced_tensors.append(reduced_tensor)
    # The tensors will be brought back to each device once they are used.
    return [reduced_tensors] * len(all_device_tensors)


class HierarchicalCopyAlgorithm(BatchAllReduceAlgorithm):
  """An algorithm that uses hierarchical copies. This is only optimized for
  eight devices connected in NetworkTopology.DGX1 or NetworkTopology.GCP_V100
  topology.
  """

  def __init__(self, network_topology):
    """Initializer for HierarchicalCopyAlgorithm.

    Args:
      network_topology: An instance of Enum class constants.NetworkTopology.
    """
    self._network_topology = network_topology

  def _do_batch_all_reduce(self, all_device_tensors):
    avail_devices = [device_tensors[0].device
                     for device_tensors in all_device_tensors]
    reduced_tensors = []
    num_devices = len(avail_devices)
    group_size = num_devices // 2
    for i, tensors_across_devices in enumerate(zip(*all_device_tensors)):
      group_0_main_device, group_1_main_device = self.__get_main_devices(
          i, num_devices)
      if group_0_main_device < group_size:
        group_0_begin = 0
        group_1_begin = group_size
      else:
        group_0_begin = group_size
        group_1_begin = 0

      # Reduce the first group.
      group_0_tensors = tensors_across_devices[group_0_begin:
                                               group_0_begin + group_size]
      with tf.device(avail_devices[group_0_main_device]):
        group_0_reduced_tensor = _all_reduce_using_copy(group_0_tensors, False)

      # Reduce the second group.
      group_1_tensors = tensors_across_devices[group_1_begin:
                                               group_1_begin + group_size]
      with tf.device(avail_devices[group_1_main_device]):
        group_1_reduced_tensor = _all_reduce_using_copy(group_1_tensors, False)

      # Reduce between the groups.
      with tf.device(avail_devices[group_0_main_device]):
        total_reduced_tensor = _all_reduce_using_copy(
            [group_0_reduced_tensor, group_1_reduced_tensor], False)

      # Broadcast the result back into the root of each group.
      with tf.device(avail_devices[group_0_main_device]):
        group_0_reduced_tensor_bcast = tf.identity(total_reduced_tensor)
      with tf.device(avail_devices[group_1_main_device]):
        group_1_reduced_tensor_bcast = tf.identity(total_reduced_tensor)

      reduced_tensors_bcast = []
      for j in range(len(tensors_across_devices)):
        with tf.device(avail_devices[j]):
          # Broadcast the result back to each member in the group from the root.
          if (group_0_main_device < group_size) == (j < group_size):
            src_device_tensor = group_0_reduced_tensor_bcast
          else:
            src_device_tensor = group_1_reduced_tensor_bcast
          reduced_tensors_bcast.append(tf.identity(src_device_tensor))

      reduced_tensors.append(reduced_tensors_bcast)

    reduced_tensors = list(zip(*reduced_tensors))
    return reduced_tensors

  def __get_main_devices(self, tensor_index, num_devices):
    """Returns the pair of main devices to use for initial reduction.

    Args:
      tensor_index: Index of the current tensor in the list of tensors to copy.
      num_devices: Total number of devices.

    Returns:
      A tuple containing pair of main device indices for the initial
      reduction. Then, the first element of the tuple should be used for the
      final reduction.

    Raises:
      ValueError: Invalid input arguments.
    """
    if self._network_topology == constants.NetworkTopology.DGX1:
      return tensor_index % num_devices, (tensor_index +
                                          (num_devices // 2)) % num_devices
    elif self._network_topology == constants.NetworkTopology.GCP_V100:
      if num_devices != 8:
        raise ValueError('HierarchicalCopy only supports eight devices in %s.' %
                         self._network_topology)
      # TODO(hinsu): Generalize main device indices to handle any other
      # isomorphic connection graph that connects two cliques using connections
      # other than 0-5 and 2-7.
      main_device_pairs = [(0, 5), (2, 7), (5, 0), (7, 2)]
      return main_device_pairs[tensor_index % len(main_device_pairs)]
    else:
      # TODO(reedwm): make this logic more general for arbitrary topology.
      raise ValueError(
          'HierarchicalCopy is not supported for %s network topology.' %
          self._network_topology)


class AllReduceSpecAlgorithm(BatchAllReduceAlgorithm):
  """An algorithm that uses an all reduce spec."""

  def __init__(self, all_reduce_spec, gpu_indices, agg_small_grads_max_bytes,
               agg_small_grads_max_group):
    spec = allreduce.parse_all_reduce_spec(all_reduce_spec)
    if len(spec) != 1:
      raise ValueError(
          'Replicated mode does not support hybrid all-reduce strategies')
    self._all_reduce_spec = spec[0]
    self._gpu_indices = gpu_indices
    self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self._agg_small_grads_max_group = agg_small_grads_max_group

  def _do_batch_all_reduce(self, all_device_tensors):
    # TODO(reedwm): Merge allreduce.sum_gradients_all_reduce with the other
    # gradient aggregation code, since gradient aggregation is doing an all
    # reduce. Currently, we do gradient repacking in two different places.
    # TODO(reedwm): Change the allreduce code to reduce tensors instead of
    # tower_grads.
    tower_grads = [[(t, None) for t in device_tensors]
                   for device_tensors in all_device_tensors]
    aggregated_device_grads = allreduce.sum_gradients_all_reduce(
        False,  # single_session
        ['/job:localhost'],
        tower_grads,
        1,
        self._all_reduce_spec.alg,
        self._all_reduce_spec.shards,
        self._gpu_indices,
        agg_small_grads_max_bytes=self._agg_small_grads_max_bytes,
        agg_small_grads_max_group=self._agg_small_grads_max_group)
    return [[t for t, _ in grad_vars] for grad_vars in aggregated_device_grads]


def algorithm_from_params(params):
  """Returns a BatchAllReduceAlgorithm from a Params tuple."""
  if params.all_reduce_spec:
    if params.gpu_indices:
      gpu_indices = [int(x) for x in params.gpu_indices.split(',')]
    else:
      gpu_indices = [x for x in range(params.num_gpus)]
    return AllReduceSpecAlgorithm(params.all_reduce_spec, gpu_indices,
                                  params.agg_small_grads_max_bytes,
                                  params.agg_small_grads_max_group)
  elif params.hierarchical_copy:
    return HierarchicalCopyAlgorithm(params.network_topology)
  else:
    if params.local_parameter_device == 'gpu':
      devices_to_reduce_on = ['/gpu:%d' % i for i in range(params.num_gpus)]
    else:
      devices_to_reduce_on = ['/cpu:0']
    #### Made only for adam optimizer ####
    return CopyToDeviceAlgorithm(devices_to_reduce_on, use_mean=True)


def _apply_to_all_device_tensors(all_device_tensors, apply_func, colocate=True):
  """Applies a function to each tensor in `all_device_tensors`.

  A new list of lists of tensors is returned, where every tensor in
  `all_device_tensors` has had `apply_func` called on it. `all_device_tensors`
  is not modified.

  Args:
    all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]` is
      a tensor where `i` is the device index and `j` is the tensor index.
    apply_func: A function taking in three arguments: tensor, device_index,
      tensor_index, and returning a modified tensor.
      `tensor` is `all_device_tensors[device_index][tensor_index]`.
    colocate: If True, apply_func will be run under context manager colocated
      with it's input tensor.
  Returns:
    A list in the same form as `all_device_tensors`, except each tensor has had
    `apply_func` called on it.
  """
  new_all_device_tensors = []
  for device_index, device_tensors in enumerate(all_device_tensors):
    new_device_tensors = []
    for tensor_index, t in enumerate(device_tensors):
      if colocate:
        with tf.colocate_with(t):
          new_t = apply_func(t, device_index, tensor_index)
      else:
        new_t = apply_func(t, device_index, tensor_index)
      new_device_tensors.append(new_t)
    new_all_device_tensors.append(new_device_tensors)
  return new_all_device_tensors


def _defer_tensor(tensor):
  """Defers the retrieval of a tensor.

  The tensor is put into a StagingArea, and the return value is the
  retrieval of the tensor from the StagingArea. The effect is that the
  tensor returned from this function is the tensor that was put in the
  StagingArea for the previous Session.run() call.

  Args:
    tensor: The tensor to defer for one step.

  Returns:
    deferred_tensor: The tensor deferred for one step.
    put_op: An op to put `tensor` in the StagingArea. Must be run every step
      that `deferred_tensor` is run.
    warmup_op: A warmup op that should be called before the first step. Puts
      a zero tensor into the StagingArea.
  """
  tensor_stage = tf.contrib.staging.StagingArea([tensor.dtype], [tensor.shape])
  put_op = tensor_stage.put([tensor])
  warmup_op = tensor_stage.put([tf.zeros(tensor.shape, dtype=tensor.dtype)])

  # Fetch the next tensor to use.
  (tensor,) = tensor_stage.get()
  return tensor, put_op, warmup_op


def _defer_all_device_tensors(all_device_tensors):
  """Defers every tensor in `all_device_tensors`."""
  put_ops = [[] for _ in all_device_tensors]
  warmup_ops = [[] for _ in all_device_tensors]
  def apply_func(tensor, device_index, tensor_index):
    del tensor_index
    tensor, put_op, warmup_op = _defer_tensor(tensor)
    put_ops[device_index].append(put_op)
    warmup_ops[device_index].append(warmup_op)
    return tensor
  new_all_device_tensors = _apply_to_all_device_tensors(all_device_tensors,
                                                        apply_func)
  return new_all_device_tensors, put_ops, warmup_ops


def _add_put_op_control_deps(all_device_tensors, num_splits, put_ops):
  """Add control dependencies from `put_ops` to `all_device_tensors`.

  This should only be called when deferred tensors are being used.

  The control dependencies are added so that the put ops are run whenever
  `all_device_tensors` is run. That way, the caller does not have to explicitly
  run the put ops.

  Args:
    all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]` is
      a tensor where `i` is the device index and `j` is the tensor index.
    num_splits: The number of splits that were used for the all-reduce.
    put_ops: A list of put ops from deferring the tensors.
  Returns:
    A list in the same form as `all_device_tensors`, except each tensor has a
    control dependency on an op in `put_ops`.

  """
  def apply_func(tensor, device_index, tensor_index):
    if num_splits == 0:
      deps = [put_ops[device_index][tensor_index]]
    else:
      deps = put_ops[device_index]
    assert len(deps) == 1
    with tf.control_dependencies(deps):
      return tf.identity(tensor, name='control_dependency')
  return _apply_to_all_device_tensors(all_device_tensors, apply_func)


def _compact_all_device_tensors(all_device_tensors):
  """Compacts each tensor by casting to fp16."""
  def apply_func(tensor, device_index, tensor_index):
    del device_index, tensor_index
    return tf.cast(tensor, tf.float16)
  return _apply_to_all_device_tensors(all_device_tensors, apply_func)


def _undo_compact_all_device_tensors(all_device_tensors,
                                     orig_all_device_tensors):
  """Uncompacts each tensor by casting to it's original dtype."""
  def apply_func(tensor, device_index, tensor_index):
    orig_tensor = orig_all_device_tensors[device_index][tensor_index]
    with tf.colocate_with(orig_tensor):
      return tf.cast(tensor, orig_tensor.dtype)
  return _apply_to_all_device_tensors(all_device_tensors, apply_func,
                                      colocate=False)


class _TensorPacker(object):
  """Packs and unpacks tensors into groups.

  This class first concatenates a set of tensors, then split the concatenated
  tensor into a small number of chunks. This is useful for all-reducing tensors,
  as doing a small number of all-reduces on large tensors can be faster than
  doing a large number of all-reduces on small tensors.
  """

  def __init__(self, num_splits):
    """Initializes the _TensorPacker.

    Args:
      num_splits: The number of tensors to split the concatenated tensor into.
        The batch all-reduce will consist of `num_splits` all-reduces.
    """
    assert num_splits > 0
    self._num_splits = num_splits
    self._next_method = 'concat'

  _concat_tensor_state = namedtuple('_concat_tensor_state',
                                    ['orig_shapes', 'orig_sizes'])

  def _concat_tensors(self, device_tensors):
    """Concatenate tensors into a single tensor."""
    flat_tensors = [tf.reshape(t, [-1]) for t in device_tensors]
    orig_shapes = [t.shape for t in device_tensors]
    orig_sizes = [s.num_elements() for s in orig_shapes]
    # All shapes must be fully defined.
    assert None not in orig_sizes
    concatenated_grad = tf.concat(flat_tensors, 0)
    return concatenated_grad, self._concat_tensor_state(orig_shapes, orig_sizes)

  def _split_tensors(self, concatenated_tensor):
    """Splits concatenated tensor into `num_splits` pieces."""
    # TODO(zhengxq): it is possible to optimize away the additional
    # data movement by copying along the original tensor boundary.
    # TODO(zhengxq): it is also possible to optimize away all the concat
    # as well.
    total_tensor_size = concatenated_tensor.shape.num_elements()
    split_size = total_tensor_size // self._num_splits
    split_size_last = total_tensor_size - split_size * (self._num_splits - 1)
    split_sizes = [split_size] * (self._num_splits - 1) + [split_size_last]
    tensor_packs = tf.split(concatenated_tensor, split_sizes)
    return tensor_packs

  def _undo_split_tensors(self, tensor_packs):
    """Undoes self._split_tensors()."""
    return tf.concat(tensor_packs, 0)

  def _undo_concat_tensors(self, concatenated_tensor, concat_tensor_state):
    """Undoes self._concat_tensors()."""
    tensors_with_sizes = tf.split(concatenated_tensor,
                                  concat_tensor_state.orig_sizes)
    tensors_with_shapes = [
        tf.reshape(grad, shape)
        for grad, shape in zip(tensors_with_sizes,
                               concat_tensor_state.orig_shapes)
    ]
    return tensors_with_shapes

  def concat_all_device_tensors(self, all_device_tensors):
    """For each device, concatenate the device's tensors into a single tensor.

    Args:
      all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]`
        is a tensor where `i` is the device index and `j` is the tensor index.

    Returns:
      A list of list of tensors in a similar form as all_device_tensors, except
      the tensors on each device have been concatenated. Each inner list
      consists of a single concatenated tensor.
    """
    assert self._next_method == 'concat'
    new_all_device_tensors = []
    tensor_states = []
    for device_tensors in all_device_tensors:
      with tf.colocate_with(device_tensors[0]):
        concat_tensor, tensor_state = self._concat_tensors(device_tensors)
        new_all_device_tensors.append([concat_tensor])
        tensor_states.append(tensor_state)
    self._tensor_states = tensor_states
    self._next_method = 'split'
    return new_all_device_tensors

  def split_all_device_tensors(self, all_device_tensors):
    """Splits concatenated tensors into `num_splits` pieces.

    `num_splits` is specified in the constructor.  In the case where the total
    size of a concatenated tensor is not divisible by `num_splits`, the last
    split tensor gets more elements.

    Args:
      all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]`
        is a tensor where `i` is the device index and `j` is the tensor index.
        For each i, `all_device_tensors[i]` must be a list of length 1 of a
        single concatenated tensor.

    Returns:
      A list of list of tensors in a similar form as all_device_tensors, except
      the concatenated tensor on each device have been split. Each inner list
      is a list of length `num_splits`.
    """
    assert self._next_method == 'split'
    new_all_device_tensors = []
    for [concat_tensor] in all_device_tensors:
      with tf.colocate_with(concat_tensor):
        new_all_device_tensors.append(self._split_tensors(concat_tensor))
    self._orig_concat_all_device_tensors = all_device_tensors
    self._next_method = 'undo_split'
    return new_all_device_tensors

  def undo_split_all_device_tensors(self, all_device_tensors):
    """Undoes the effects of `split_all_device_tensors`."""
    assert self._next_method == 'undo_split'
    new_all_device_tensors = []
    for i, device_tensors in enumerate(all_device_tensors):
      [orig_tensor] = self._orig_concat_all_device_tensors[i]
      with tf.colocate_with(orig_tensor):
        new_all_device_tensors.append(
            [self._undo_split_tensors(device_tensors)])
    self._next_method = 'undo_concat'
    return new_all_device_tensors

  def undo_concat_all_device_tensors(self, all_device_tensors):
    """Undoes the effects of `concat_all_device_tensors`."""
    assert self._next_method == 'undo_concat'
    new_all_device_tensors = []
    for [concat_tensor], tensor_state in zip(all_device_tensors,
                                             self._tensor_states):
      with tf.colocate_with(concat_tensor):
        new_all_device_tensors.append(self._undo_concat_tensors(concat_tensor,
                                                                tensor_state))
    self._next_method = None
    return new_all_device_tensors
