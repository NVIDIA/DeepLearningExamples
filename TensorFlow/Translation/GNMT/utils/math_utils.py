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

"""Generally useful utility functions."""
from __future__ import print_function

import collections
import six
import os

import tensorflow as tf

from tensorflow.python.framework import function
from tensorflow.python.ops import gen_nn_ops


def sparse_softmax_crossent_with_logits(logits=None, labels=None, name=None):
  """docstring."""
  # TODO(jamesqin): merge with tf.nn.sparse_softmax_cross_entropy_with_logits
  # Basically forks the tf lib function, only that the result isn't casted
  # back to tf.float16 if the input is tf.float16
  # TODO(jamesqin): implement a fused kernel to reduce memory footprint.

  # Reshape logits and labels to rank 2.
  with tf.name_scope(name, "SparseSoftmaxCrossEntropyWithLogits",
                     [labels, logits]):
    labels = tf.convert_to_tensor(labels)
    logits = tf.convert_to_tensor(logits)
    precise_logits = tf.cast(logits, tf.float32) if (tf.as_dtype(
        logits.dtype) == tf.float16) else logits

    # Store label shape for result later.
    labels_static_shape = labels.get_shape()
    labels_shape = tf.shape(labels)
    static_shapes_fully_defined = (
        labels_static_shape.is_fully_defined() and
        logits.get_shape()[:-1].is_fully_defined())
    if logits.get_shape().ndims is not None and logits.get_shape().ndims == 0:
      raise ValueError(
          "Logits cannot be scalars - received shape %s." % logits.get_shape())
    if logits.get_shape().ndims is not None and (
        labels_static_shape.ndims is not None and
        labels_static_shape.ndims != logits.get_shape().ndims - 1):
      raise ValueError("Rank mismatch: Rank of labels (received %s) should "
                       "equal rank of logits minus 1 (received %s)." %
                       (labels_static_shape.ndims, logits.get_shape().ndims))
    if (static_shapes_fully_defined and
        labels_static_shape != logits.get_shape()[:-1]):
      raise ValueError("Shape mismatch: The shape of labels (received %s) "
                       "should equal the shape of logits except for the last "
                       "dimension (received %s)." % (labels_static_shape,
                                                     logits.get_shape()))
    # Check if no reshapes are required.
    if logits.get_shape().ndims == 2:
      cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
          precise_logits, labels, name=name)
      # cost.dtype is always fp32
      return cost

    # Perform a check of the dynamic shapes if the static shapes are not fully
    # defined.
    shape_checks = []
    if not static_shapes_fully_defined:
      xla_compile = (os.environ["xla_compile"] == "true")
      use_xla = (os.environ["use_xla"] == "true")
      if not (xla_compile or use_xla):
        # Assert isn't registered w/ GPU, not working w/ xla.compile()
        shape_checks.append(
            tf.assert_equal(
                tf.shape(labels),
                tf.shape(logits)[:-1]))
    with tf.control_dependencies(shape_checks):
      # Reshape logits to 2 dim, labels to 1 dim.
      num_classes = tf.shape(logits)[tf.rank(logits) - 1]
      precise_logits = tf.reshape(precise_logits, [-1, num_classes])
      labels = tf.reshape(labels, [-1])
      # The second output tensor contains the gradients.  We use it in
      # _CrossEntropyGrad() in nn_grad but not here.
      cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
          precise_logits, labels, name=name)
      cost = tf.reshape(cost, labels_shape)
      cost.set_shape(labels_static_shape)
      # cost is always fp32
      return cost


def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):
  """Custom version of tf.clip_by_global_norm that doesn't check numerics."""
  if (not isinstance(t_list, collections.Sequence)
      or isinstance(t_list, six.string_types)):
    raise TypeError("t_list should be a sequence")
  t_list = list(t_list)
  if use_norm is None:
    use_norm = tf.global_norm(t_list, name)

  with tf.name_scope(name, "clip_by_global_norm", t_list + [clip_norm]) as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale = clip_norm * tf.minimum(
        1.0 / use_norm,
        tf.constant(1.0, dtype=use_norm.dtype) / clip_norm)

    values = [
        tf.convert_to_tensor(
            t.values if isinstance(t, tf.IndexedSlices) else t,
            name="t_%d" % i)
        if t is not None else t
        for i, t in enumerate(t_list)]

    values_clipped = []
    for i, v in enumerate(values):
      if v is None:
        values_clipped.append(None)
      else:
        with tf.colocate_with(v):
          values_clipped.append(
              tf.identity(v * scale, name="%s_%d" % (name, i)))

    list_clipped = [
        tf.IndexedSlices(c_v, t.indices, t.dense_shape)
        if isinstance(t, tf.IndexedSlices)
        else c_v
        for (c_v, t) in zip(values_clipped, t_list)]

  return list_clipped, use_norm


def BatchMatMul(a, b):
  use_fp32_batch_matmul = (os.environ["use_fp32_batch_matmul"] == "true")
  xla_compile = (os.environ["xla_compile"] == "true")
  if use_fp32_batch_matmul:
    def DoFn(a, b):
      dtype = a.dtype
      a = tf.to_float(a)
      b = tf.to_float(b)
      return tf.cast(tf.matmul(a, b), dtype)
    # If using xla_compile, the fwd and bak per tower are wrapped in xla_compile
    if not xla_compile:
      DoFn = function.Defun(noinline=True)(DoFn)
      res = DoFn(a, b)
      res.set_shape((None, None, b.shape[-1].value))
    else:
      # If xla_compile, leave to xla to handle the casts.
      res = DoFn(a, b)
  else:
    res = tf.matmul(a, b)
  return res


