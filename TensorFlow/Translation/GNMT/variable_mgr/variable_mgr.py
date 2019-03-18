# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Defines VariableMgr and subclasses used to manage variables.

"""

from __future__ import print_function

import re

import tensorflow as tf

from utils import misc_utils
from variable_mgr import allreduce
from variable_mgr import batch_allreduce
from variable_mgr import variable_mgr_util


class VariableMgr(object):
  """Abstract superclass for class used by BenchmarkCNN to control variables.

    Functions on this class are used to control how variables are created and
    managed, and how gradients are computed and applied.
  """

  def __init__(self, benchmark_cnn):
    self.benchmark_cnn = benchmark_cnn
    self.staging_delta_ops = []
    self.use_resource_vars = benchmark_cnn.params.use_resource_vars

    # A variable for automatic loss scaling.
    self.grad_has_inf_nan = None

  def each_tower_has_variables(self):
    """Returns True if each GPU tower of the model has separate variables."""
    assert False, 'Must be implemented in subclass'

  def supports_staged_vars(self):
    """Whether staged variable management is supported."""
    return False

  def create_outer_variable_scope(self, device_num):
    """Create the tf.variable_scope around all model graph operations."""
    del device_num  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def preprocess_device_grads(self, device_grads):
    """Preprocess the device gradients prior to applying them.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.

    Returns: a tuple of (apply_gradients_devices, gradient_state).
      gradient_state is an opaque structure that should be passed to
      get_gradients_to_apply() and append_apply_gradients_ops() (in that order).
      apply_gradients_devices is a list of devices where the gradients will be
      applied with get_gradients_to_apply() and append_apply_gradients_ops().
    """
    del device_grads  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def get_gradients_to_apply(self, device_num, gradient_state):
    """Returns the [(gradient, variable)] list to apply for device_num.

    Args:
      device_num: indexes into apply_gradients_devices, which was returned by an
        earlier call to preprocess_device_grads.
      gradient_state: from previous call to apply_gradients_devices.
    """
    del device_num, gradient_state  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops,
                                 loss_scale_params):
    """Adds training ops for grads to 'training_ops'.



    Args:
      gradient_state: from previous call to apply_gradients_devices.
      opt: the underlying optimizer
      grads: [(grad, var)] to apply
      training_ops: list to which to add ops
      loss_scale_params: parameters for loss scaling.
    """
    del gradient_state  # unused by this implementation

    def get_apply_gradients_ops_func():
      """Returns the apply_gradients op."""
      return [opt.apply_gradients(grads)]

    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops, get_apply_gradients_ops_func, loss_scale_params,
        self.grad_has_inf_nan)

  def get_post_init_ops(self):
    """Returns ops that should run post-initialization."""
    return []

  def get_devices(self):
    """Returns devices to use for computation; includes replica selection."""
    assert False, 'Must be implemented in subclass'

  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    return tf.global_variables()

  def trainable_variables_on_device(self,
                                    rel_device_num,
                                    abs_device_num,
                                    writable=False):
    """Return the set of trainable variables on device.

    Args:
      rel_device_num: local worker device index.
      abs_device_num: global graph device index.
      writable: whether to get a reference to the underlying variable.

    Returns:
      The set of trainable variables on the specified device.
    """
    del rel_device_num, writable
    if self.each_tower_has_variables():
      params = [
          v for v in tf.trainable_variables()
          if v.name.startswith('v%s/' % abs_device_num)
      ]
    else:
      params = tf.trainable_variables()
    return params

class VariableMgrLocalReplicated(VariableMgr):
  """VariableMgr that implements the --replicated mode for local jobs.

     Each GPU has its own copy of the variables. To apply gradients,
     either a local all-reduce algorithm is applied or a regular
     cross-device aggregation is used to replicate the combined
     gradients to all towers.
  """

  def __init__(self, benchmark_cnn, all_reduce_spec,
               agg_small_grads_max_bytes, agg_small_grads_max_group,
               allreduce_merge_scope):
    super(VariableMgrLocalReplicated, self).__init__(benchmark_cnn)
    if all_reduce_spec:
      spec = allreduce.parse_all_reduce_spec(all_reduce_spec)
      if len(spec) != 1:
        raise ValueError(
            'replicated mode does not support hybrid all-reduce strategies')
      self._all_reduce_spec = spec[0]
    else:
      self._all_reduce_spec = None
    self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self._agg_small_grads_max_group = agg_small_grads_max_group
    self._warmup_ops = []
    self._allreduce_merge_scope = allreduce_merge_scope
    self._gradient_put_ops = None

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v%s' % device_num,
                             use_resource=self.use_resource_vars)

  def preprocess_device_grads(self, device_grads):
    compact_grads = (self.benchmark_cnn.params.use_fp16 and
                     self.benchmark_cnn.params.compact_gradient_transfer)
    defer_grads = (self.benchmark_cnn.params.variable_consistency == 'relaxed')

    grads_to_reduce = [[g for g, _ in grad_vars] for grad_vars in device_grads]
    algorithm = batch_allreduce.algorithm_from_params(self.benchmark_cnn.params)
    reduced_grads, self._warmup_ops = algorithm.batch_all_reduce(
        grads_to_reduce, self.benchmark_cnn.params.gradient_repacking,
        compact_grads, defer_grads)
    assert not self._warmup_ops
    if (self.benchmark_cnn.params.use_fp16 and
        self.benchmark_cnn.enable_auto_loss_scale):
      # Check for infs or nans
      is_finite_list = []
      with tf.name_scope('check_for_inf_and_nan'):
        for tower_grads in reduced_grads:
          with tf.colocate_with(tower_grads[0]):
            # TODO(tanmingxing): Create fused op that takes in a list of tensors
            # as input and returns scalar boolean True if there are any
            # infs/nans.
            is_finite_list.append(tf.reduce_all(
                [tf.reduce_all(tf.is_finite(g)) for g in tower_grads]))
        self.grad_has_inf_nan = tf.logical_not(tf.reduce_all(is_finite_list))
    reduced_device_grads = [[
        (g, v) for g, (_, v) in zip(grads, grad_vars)
    ] for grads, grad_vars in zip(reduced_grads, device_grads)]
    return self.benchmark_cnn.devices, reduced_device_grads

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    return device_grads[device_num]

  def get_post_init_ops(self):
    # Copy initialized values for variables on GPU 0 to other GPUs.
    global_vars = tf.global_variables()
    var_by_name = dict([(v.name, v) for v in global_vars])
    post_init_ops = []
    copy_froms = set()
    skipped_vars = []
    for v in global_vars:
      split_name = v.name.split('/')
      # TODO(b/62630508): use more specific prefix than v or v0.
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        skipped_vars.append(v)
        continue
      # Only vars starts with "v[number]" are synced.
      split_name[0] = 'v0'
      copy_from = var_by_name['/'.join(split_name)]
      copy_froms.add(copy_from)
      post_init_ops.append(v.assign(copy_from.read_value()))
    post_init_ops += self._warmup_ops
    # If copy-froms is empty, then all vars are actually saved.
    misc_utils.print_out('All copy-from vars(%d): ' % len(copy_froms))
    for gv in copy_froms:
      misc_utils.print_out(gv.name)
    misc_utils.print_out('All skippped vars(%d): ' % len(skipped_vars))
    for gv in skipped_vars:
      misc_utils.print_out(gv.name)
    assert len(skipped_vars) >= len(copy_froms)

    return post_init_ops

  def savable_variables(self):
    """Return the set of variables used for saving/loading the model."""
    params = []
    for v in tf.global_variables():
      split_name = v.name.split('/')
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        params.append(v)
    return params

  def get_devices(self):
    return self.benchmark_cnn.raw_devices

