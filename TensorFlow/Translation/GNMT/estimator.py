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
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
"""Estimator functions supporting running on TPU."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import subprocess
import time
import numpy as np
import tensorflow as tf

from tensorflow.contrib.compiler import xla
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.util import tf_contextlib

import gnmt_model
import model_helper
from utils import iterator_utils
from utils import misc_utils
from utils import nmt_utils
from utils import vocab_utils
from variable_mgr import variable_mgr
from variable_mgr import variable_mgr_util

from benchmark_hooks import BenchmarkHook


def _get_custom_getter():
  """Returns a custom getter that this class's methods must be called under.

  All methods of this class must be called under a variable scope that was
  passed this custom getter. Example:

  ```python
  network = ConvNetBuilder(...)
  with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
    network.conv(...)
    # Call more methods of network here
  ```

  Currently, this custom getter only does anything if self.use_tf_layers is
  True. In that case, it causes variables to be stored as dtype
  self.variable_type, then casted to the requested dtype, instead of directly
  storing the variable as the requested dtype.
  """

  def inner_custom_getter(getter, *args, **kwargs):
    """Custom getter that forces variables to have type self.variable_type."""
    cast_to_float16 = False
    requested_dtype = kwargs["dtype"]
    if requested_dtype == tf.float16:
      # Only change the variable dtype if doing so does not decrease variable
      # precision.
      kwargs["dtype"] = tf.float32
      cast_to_float16 = True
    var = getter(*args, **kwargs)
    with tf_ops.init_scope():
      # This if statement is needed to guard the cast, because batch norm
      # assigns directly to the return value of this custom getter. The cast
      # makes the return value not a variable so it cannot be assigned. Batch
      # norm variables are always in fp32 so this if statement is never
      # triggered for them.
      if cast_to_float16:
        var = tf.cast(var, tf.float16)
    return var

  return inner_custom_getter


@tf_contextlib.contextmanager
def mixed_precision_scope():
  with tf.variable_scope("", custom_getter=_get_custom_getter()) as varscope:
    yield varscope


def maybe_xla_compile(hparams, fn, *args):
  pure_fn = lambda: fn(*args)
  if hparams and hparams.xla_compile:
    return xla.compile(pure_fn)
  else:
    return pure_fn()


class ModelFnFactory(object):
  """docstring."""

  def __init__(self, hparams):
    self.hparams = hparams

  def build_graph_dist_strategy(self, features, labels, mode, params):
    """Model function."""
    del labels, params
    misc_utils.print_out("Running dist_strategy mode_fn")

    hparams = self.hparams

    # Create a GNMT model for training.
    # assert (hparams.encoder_type == "gnmt" or
    #        hparams.attention_architecture in ["gnmt", "gnmt_v2"])
    with mixed_precision_scope():
      model = gnmt_model.GNMTModel(hparams, mode=mode, features=features)
      if mode == tf.contrib.learn.ModeKeys.INFER:
        sample_ids = model.sample_id
        reverse_target_vocab_table = lookup_ops.index_to_string_table_from_file(
            hparams.tgt_vocab_file, default_value=vocab_utils.UNK)
        sample_words = reverse_target_vocab_table.lookup(
            tf.to_int64(sample_ids))
        # make sure outputs is of shape [batch_size, time] or [beam_width,
        # batch_size, time] when using beam search.
        if hparams.time_major:
          sample_words = tf.transpose(sample_words)
        elif sample_words.shape.ndims == 3:
          # beam search output in [batch_size, time, beam_width] shape.
          sample_words = tf.transpose(sample_words, [2, 0, 1])
        predictions = {"predictions": sample_words}
        # return loss, vars, grads, predictions, train_op, scaffold
        return None, None, None, predictions, None, None
      elif mode == tf.contrib.learn.ModeKeys.TRAIN:
        loss = model.train_loss
        train_op = model.update
        return loss, model.params, model.grads, None, train_op, None
      else:
        raise ValueError("Unknown mode in model_fn: %s" % mode)

  def _create_loss_scale_vars(self):
    """docstring."""
    # Create loss scale vars if necessary
    hparams = self.hparams
    loss_scale, loss_scale_normal_steps = None, None
    if hparams.use_fp16:
      loss_scale = tf.get_variable(
          "loss_scale",
          initializer=float(hparams.fp16_loss_scale),
          dtype=tf.float32,
          trainable=False)
      if hparams.enable_auto_loss_scale:
        loss_scale_normal_steps = tf.get_variable(
            "loss_scale_normal_steps", initializer=0, trainable=False)

    return loss_scale, loss_scale_normal_steps

  def _shard_inputs(self, features, num_towers):
    """docstring."""
    if num_towers == 1:
      return [features]

    source = features["source"]
    target_input = features["target_input"]
    target_output = features["target_output"]
    source_seq_length = features["source_sequence_length"]
    target_seq_length = features["target_sequence_length"]

    # Compute each split sizes.
    global_batch_size = tf.size(source_seq_length)
    tower_batch_size = tf.cast(global_batch_size / num_towers, dtype=tf.int32)

    split_sizes = [tower_batch_size] * (num_towers - 1)
    split_sizes.append(global_batch_size - (num_towers - 1) * tower_batch_size)

    sources = tf.split(source, split_sizes, axis=0)
    target_inputs = tf.split(target_input, split_sizes, axis=0)
    target_outputs = tf.split(target_output, split_sizes, axis=0)
    source_sequence_lengths = tf.split(source_seq_length, split_sizes)
    target_sequence_lengths = tf.split(target_seq_length, split_sizes)

    tower_features = []
    for i in range(num_towers):
      tower_features.append({
          "source": sources[i],
          "target_input": target_inputs[i],
          "target_output": target_outputs[i],
          "source_sequence_length": source_sequence_lengths[i],
          "target_sequence_length": target_sequence_lengths[i]
      })

    return tower_features

  def get_optimizer(self, hparams, learning_rate):
    """docstring."""
    if hparams.optimizer == "sgd":
      opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif hparams.optimizer == "adam":
      opt = tf.train.AdamOptimizer(learning_rate)
    else:
      raise ValueError("Unknown optimizer type %s" % hparams.optimizer)
    return opt

  def _compute_tower_grads(self, tower_loss, tower_params, learning_rate, use_fp16=False,
                           loss_scale=None, colocate_gradients_with_ops=True):
    """docstring."""
    if use_fp16:
      assert loss_scale
      scaled_loss = tf.multiply(
          tower_loss,
          tf.convert_to_tensor(loss_scale, dtype=tower_loss.dtype),
          name="scaling_loss")
    else:
      scaled_loss = tower_loss

    opt = self.get_optimizer(self.hparams, learning_rate)
    grads_and_vars = opt.compute_gradients(scaled_loss, tower_params,
            colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)
    grads = [x for (x, _) in grads_and_vars]
    assert grads
    for g in grads:
      assert g.dtype == tf.float32, "grad.dtype isn't fp32: %s" % g.name
    # Downscale grads
    for var, grad in zip(tower_params, grads):
      if grad is None:
        misc_utils.print_out("%s gradient is None!" % var.name)

    if use_fp16:
      grads = [
          grad * tf.reciprocal(loss_scale) for grad in grads
      ]
    return tower_params, grads, opt

  def _get_variable_mgr(self, hparams):
    """docstring."""
    assert not hparams.use_dist_strategy

    # A hack to create a container object that later get passed to VariableMgr
    # __init__() as the ill-designed `benchmark_cnn` argument.
    class Config(object):
      pass

    config = Config()
    config.params = Config()
    params = config.params
    # This is num_gpus per worker, a.k.a the number of towers.
    params.num_gpus = hparams.num_gpus
    # TODO(jamesqin): make more robust
    params.use_resource_vars = hparams.use_resource_vars
    params.use_fp16 = hparams.use_fp16
    params.compact_gradient_transfer = hparams.compact_gradient_transfer
    # For nmt, only strong consistency
    params.variable_consistency = "strong"
    params.all_reduce_spec = hparams.all_reduce_spec
    params.gpu_indices = hparams.gpu_indices
    params.agg_small_grads_max_bytes = hparams.agg_small_grads_max_bytes
    params.agg_small_grads_max_group = hparams.agg_small_grads_max_group
    params.hierarchical_copy = hparams.hierarchical_copy
    params.network_topology = hparams.network_topology
    params.local_parameter_device = hparams.local_parameter_device
    params.gradient_repacking = hparams.gradient_repacking
    params.allreduce_merge_scope = hparams.allreduce_merge_scope

    config.enable_auto_loss_scale = hparams.enable_auto_loss_scale
    if hparams.num_gpus > 0:
      config.raw_devices = ["gpu:%i" % i for i in range(hparams.num_gpus)]
    else:
      config.raw_devices = ["cpu:0"]
    config.devices = config.raw_devices

    return variable_mgr.VariableMgrLocalReplicated(
        config, config.params.all_reduce_spec,
        config.params.agg_small_grads_max_bytes,
        config.params.agg_small_grads_max_group,
        config.params.allreduce_merge_scope)

  def _print_varinfo(self, var_params, tower_id):
    # Print trainable variables
    misc_utils.print_out("# Trainable variables for tower: %d" % tower_id)
    misc_utils.print_out(
        "Format: <name>, <shape>, <dtype>, <(soft) device placement>")
    for param in var_params:
      misc_utils.print_out(
          "  %s, %s, %s, %s" % (param.name, str(param.get_shape()),
                                param.dtype.name, param.op.device))
    misc_utils.print_out("Total params size: %.2f GB" % (4. * np.sum([
        p.get_shape().num_elements()
        for p in var_params
        if p.get_shape().is_fully_defined()
    ]) / 2**30))

  def build_graph(self, features, labels, mode, params):
    """docstring."""
    del labels, params
    misc_utils.print_out("Running fast mode_fn")

    hparams = self.hparams

    # Create global_step
    tf.train.get_or_create_global_step()

    if mode == tf.contrib.learn.ModeKeys.INFER:
      # Doing inference only on one GPU
      inf_hparams = tf.contrib.training.HParams(**hparams.values())
      inf_hparams.set_hparam("num_gpus", 1)
      # Inference is done in fp32 and in the same way as that of dist_strategy.
      inf_hparams.set_hparam("use_fp16", False)

      misc_utils.print_out("inference hparmas:")
      misc_utils.print_hparams(inf_hparams)

      # Create variable_mgr
      var_mgr = self._get_variable_mgr(inf_hparams)

      with mixed_precision_scope(), tf.device("gpu:0"), tf.name_scope(
          "tower_0"), var_mgr.create_outer_variable_scope(0):
        model = gnmt_model.GNMTModel(inf_hparams, mode=mode, features=features)
        sample_ids = model.sample_id
        reverse_target_vocab_table = lookup_ops.index_to_string_table_from_file(
            inf_hparams.tgt_vocab_file, default_value=vocab_utils.UNK)
        sample_words = reverse_target_vocab_table.lookup(
            tf.to_int64(sample_ids))
        # make sure outputs is of shape [batch_size, time] or [beam_width,
        # batch_size, time] when using beam search.
        if inf_hparams.time_major:
          sample_words = tf.transpose(sample_words)
        elif sample_words.shape.ndims == 3:
          # beam search output in [batch_size, time, beam_width] shape.
          sample_words = tf.transpose(sample_words, [2, 0, 1])
        predictions = {"predictions": sample_words}
        # return loss, vars, grads, predictions, train_op, scaffold
        return None, None, None, predictions, None, None
    elif mode == tf.contrib.learn.ModeKeys.TRAIN:
      num_towers = hparams.num_gpus
      # Shard inputs
      tower_features = self._shard_inputs(features, num_towers)
      # Create loss scale vars if necessary
      loss_scale, loss_scale_normal_steps = self._create_loss_scale_vars()

      # Create variable_mgr
      var_mgr = self._get_variable_mgr(hparams)

      # Build per-tower fprop and bprop
      devices = var_mgr.get_devices()
      tower_gradvars = []
      tower_scopes = []
      var_scopes = []
      train_losses = []
      learning_rates = []
      batch_sizes = []
      opts = []

      def fprop_and_bprop(tid):
        """docstring."""
        model = gnmt_model.GNMTModel(
            hparams, mode=mode, features=tower_features[tid])
        # sync training.
        assert model.learning_rate is not None
        # The following handles shouldn't be built in when doing manual
        assert model.grad_norm is None
        assert model.update is None
        tower_loss = model.train_loss
        # Only check loss numerics if in fp16
        if hparams.use_fp16 and hparams.check_tower_loss_numerics:
          tower_loss = tf.check_numerics(
              tower_loss, "tower_%d has Inf/NaN loss" % tid)
        # Cast to fp32, otherwise would easily overflow.
        tower_loss = tf.to_float(tower_loss)
        var_params, grads, opt = self._compute_tower_grads(
            tower_loss,
            var_mgr.trainable_variables_on_device(tid, tid),
            model.learning_rate,
            use_fp16=hparams.use_fp16,
            loss_scale=loss_scale,
            colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)
        self._print_varinfo(var_params, tid)
        res = [model.train_loss, model.learning_rate, model.batch_size]
        res.extend(grads)
        opts.append(opt)
        return res

      def unpack_fprop_and_bprop_output(output):
        train_loss = output[0]
        learning_rate = output[1]
        batch_size = output[2]
        grads = output[3:]
        return train_loss, learning_rate, batch_size, grads

      with mixed_precision_scope():
        for tid in range(num_towers):
          with tf.device(devices[tid % len(devices)]), tf.name_scope(
              "tower_%s" % tid) as scope:
            tower_scopes.append(scope)
            with var_mgr.create_outer_variable_scope(tid) as var_scope:
              var_scopes.append(var_scope)

              outputs = maybe_xla_compile(hparams, fprop_and_bprop, tid)
              (train_loss, learning_rate, batch_size,
               grads) = unpack_fprop_and_bprop_output(outputs)
              train_losses.append(train_loss)
              learning_rates.append(learning_rate)
              batch_sizes.append(batch_size)
              var_params = var_mgr.trainable_variables_on_device(tid, tid)
              tower_gradvars.append(list(zip(grads, var_params)))

      # Add summaries
      if hparams.show_metrics:
        tf.summary.scalar("learning_rate", learning_rates[0])
        if loss_scale:
          tf.summary.scalar("loss_scale", loss_scale)
          if hparams.enable_auto_loss_scale:
            tf.summary.scalar("loss_scale_normal_steps",
                              loss_scale_normal_steps)
      misc_utils.print_out("Finish building fprop and per-tower bprop.")
      # Aggregate gradients
      # The following compute the aggregated grads for each tower, stored in
      # opaque grad_states structure.
      apply_grads_devices, grad_states = var_mgr.preprocess_device_grads(
          tower_gradvars)
      master_grads = None
      master_params = None
      update_ops = []
      for i, device in enumerate(apply_grads_devices):
        with tf.device(device), tf.name_scope(tower_scopes[i]):
          # Get per-tower grads.
          with tf.name_scope("get_gradients_to_apply"):
            avg_gradvars = var_mgr.get_gradients_to_apply(i, grad_states)
          avg_grads = [gv[0] for gv in avg_gradvars]

          # gradients post-processing
          with tf.name_scope("clip_gradients"):
            if hparams.clip_grads:
              clipped_grads, grad_norm = model_helper.gradient_clip(
                  avg_grads, max_gradient_norm=hparams.max_gradient_norm)
              # summary the grad on the 1st tower
              if i == 0 and hparams.show_metrics:
                tf.summary.scalar("grad_norm", grad_norm)
                tf.summary.scalar("clipped_grad_norm",
                                  tf.global_norm(clipped_grads))
            else:
              clipped_grads = avg_grads
            if i == 0:
              master_grads = clipped_grads

          # Build apply-gradients ops
          clipped_gradvars = list(
              zip(clipped_grads, [gv[1] for gv in avg_gradvars]))
          if i == 0:
            master_params = [gv[1] for gv in avg_gradvars]
          with tf.name_scope("append_gradient_ops"):
            loss_scale_params = variable_mgr_util.AutoLossScaleParams(
                enable_auto_loss_scale=hparams.enable_auto_loss_scale,
                loss_scale=loss_scale,
                loss_scale_normal_steps=loss_scale_normal_steps,
                inc_loss_scale_every_n=hparams.fp16_inc_loss_scale_every_n,
                is_chief=True)
            opt = opts[i]
            var_mgr.append_apply_gradients_ops(grad_states, opt,
                                               clipped_gradvars, update_ops,
                                               loss_scale_params)
      misc_utils.print_out("Finish building grad aggregation.")

      assert len(update_ops) == num_towers
      train_op = tf.group(update_ops)
      with tf.control_dependencies([train_op]):
        global_step = tf.train.get_global_step()
        train_op = global_step.assign_add(1)

      # Compute loss on the first gpu
      # TODO(jamesqin): optimize it?
      with tf.device("gpu:0"):
        loss = misc_utils.weighted_avg(train_losses, batch_sizes)

      # Create local init_ops
      # TODO(jamesqin): handle resource variables!
      # At present if not using mirror strategy, not using resource vars.
      local_init_ops = []
      local_init_op = tf.local_variables_initializer()
      with tf.control_dependencies([local_init_op]):
        local_init_ops.append(var_mgr.get_post_init_ops())
      local_init_ops.extend([local_init_op, tf.tables_initializer()])

      saveable_vars = var_mgr.savable_variables()
      # Add saveables for cudnn vars in master tower.
      saveable_objects = tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
      saveable_objects = [x for x in saveable_objects if "v0" in x.name]

      misc_utils.print_out("Saveable vars(%d): " % len(saveable_vars))
      for mv in saveable_vars:
        misc_utils.print_out(mv.name)

      misc_utils.print_out(
          "All global trainable vars(%d): " % len(tf.trainable_variables()))
      for tv in tf.trainable_variables():
        misc_utils.print_out(tv.name)

      misc_utils.print_out(
          "All global vars(%d): " % len(tf.global_variables()))
      for gv in tf.global_variables():
        misc_utils.print_out(gv.name)

      misc_utils.print_out(
          "master backproped params(%d): " % len(master_params))
      for mp in master_params:
        misc_utils.print_out(mp.name)

      # Note the cudnn vars are skipped the init check. :(
      scaffold = tf.train.Scaffold(
          ready_op=tf.report_uninitialized_variables(saveable_vars),
          ready_for_local_init_op=tf.report_uninitialized_variables(
              saveable_vars),
          local_init_op=tf.group(*local_init_ops),
          saver=tf.train.Saver(saveable_vars + saveable_objects, save_relative_paths=True))

      misc_utils.print_out("Finish building model_fn")
      # return loss, vars, grads, predictions, train_op, scaffold
      return loss, master_params, master_grads, None, train_op, scaffold


def make_model_fn(hparams):
  """Construct a GNMT model function for training."""
  factory = ModelFnFactory(hparams)

  if hparams.use_dist_strategy:
    def fn(features, labels, mode, params):
      """docstring."""
      (loss, _, _, predictions, train_op,
       _) = factory.build_graph_dist_strategy(features, labels, mode, params)
      if mode == tf.contrib.learn.ModeKeys.INFER:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
      else:
        if hparams.use_tpu:
          return tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode, loss=loss, train_op=train_op)
        else:
          return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                            train_op=train_op)
    return fn
  else:
    build_fn = factory.build_graph
    def fn(features, labels, mode, params):
      """docstring."""
      (loss, _, _, predictions, train_op, scaffold) = build_fn(
          features, labels, mode, params)
      if mode == tf.contrib.learn.ModeKeys.INFER:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
      else:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          scaffold=scaffold,
                                          train_op=train_op)
    return fn


def make_input_fn(hparams, mode):
  """Construct a input function for training."""

  def _input_fn(params):
    """Input function."""
    del params

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
      tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    else:
      if hparams.mode == "translate":
        src_file = hparams.translate_file + ".tok"
        tgt_file = hparams.translate_file + ".tok"
      else:
        src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      # Run one epoch and stop if running train_and_eval.
      if hparams.mode == "train_and_eval":
        # In this mode input pipeline is restarted every epoch, so choose a
        # different random_seed.
        num_repeat = 1
        random_seed = hparams.random_seed + int(time.time()) % 100
      else:
        num_repeat = 8
        random_seed = hparams.random_seed
      return iterator_utils.get_iterator(
          src_dataset,
          tgt_dataset,
          src_vocab_table,
          tgt_vocab_table,
          batch_size=hparams.batch_size,
          sos=hparams.sos,
          eos=hparams.eos,
          random_seed=random_seed,
          num_buckets=hparams.num_buckets,
          src_max_len=hparams.src_max_len,
          tgt_max_len=hparams.tgt_max_len,
          output_buffer_size=None,
          skip_count=None,
          num_shards=1,  # flags.num_workers
          shard_index=0,  # flags.jobid
          reshuffle_each_iteration=True,
          use_char_encode=hparams.use_char_encode,
          num_repeat=num_repeat,
          filter_oversized_sequences=True
      )  # need to update get_effective_train_epoch_size() if this flag flips.
    else:
      return iterator_utils.get_infer_iterator(
          src_dataset,
          src_vocab_table,
          batch_size=hparams.infer_batch_size,
          eos=hparams.eos,
          src_max_len=hparams.src_max_len,
          use_char_encode=hparams.use_char_encode)

  def _synthetic_input_fn(params):
    """Fake inputs for debugging and benchmarking."""
    del params
    batch_size = hparams.batch_size
    src_max_len = hparams.src_max_len
    tgt_max_len = hparams.tgt_max_len
    features = {
        "source":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=1,
                shape=(batch_size, src_max_len)),
        "target_input":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=2,
                shape=(batch_size, tgt_max_len)),
        "target_output":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=3,
                shape=(batch_size, tgt_max_len)),
        "source_sequence_length":
            tf.constant([src_max_len] * batch_size),
        "target_sequence_length":
            tf.constant([tgt_max_len] * batch_size)
    }
    return features

  if hparams.use_synthetic_data:
    return _synthetic_input_fn
  else:
    return _input_fn


def get_distribution_strategy(num_gpus):
  if num_gpus == 0:
    return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def get_sacrebleu(trans_file, detokenizer_file):
  """Detokenize the trans_file and get the sacrebleu score."""
  assert tf.gfile.Exists(detokenizer_file)
  local_detokenizer_file = "/tmp/detokenizer.perl"
  if tf.gfile.Exists(local_detokenizer_file):
    tf.gfile.Remove(local_detokenizer_file)
  tf.gfile.Copy(detokenizer_file, local_detokenizer_file, overwrite=True)

  assert tf.gfile.Exists(trans_file)
  local_trans_file = "/tmp/newstest2014_out.tok.de"
  if tf.gfile.Exists(local_trans_file):
    tf.gfile.Remove(local_trans_file)
  tf.gfile.Copy(trans_file, local_trans_file, overwrite=True)

  detok_trans_path = "/tmp/newstest2014_out.detok.de"
  if tf.gfile.Exists(detok_trans_path):
    tf.gfile.Remove(detok_trans_path)

  # Detokenize the trans_file.
  cmd = "cat %s | perl %s -l de | cat > %s" % (
      local_trans_file, local_detokenizer_file, detok_trans_path)
  subprocess.run(cmd, shell=True)
  assert tf.gfile.Exists(detok_trans_path)

  # run sacrebleu
  cmd = ("cat %s | sacrebleu -t wmt14/full -l en-de --score-only -lc --tokenize"
         " intl") % (detok_trans_path)
  sacrebleu = subprocess.run([cmd], stdout=subprocess.PIPE, shell=True)
  score = sacrebleu.stdout.strip()
  return float(score)


def get_metrics(hparams, model_fn, ckpt=None, only_translate=False):
  """Run inference and compute metrics."""
  pred_estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=hparams.output_dir)

  benchmark_hook = BenchmarkHook(hparams.infer_batch_size)

  predictions = pred_estimator.predict(
      make_input_fn(hparams, tf.contrib.learn.ModeKeys.INFER),
      checkpoint_path=ckpt, hooks=[benchmark_hook])
  translations = []
  output_tokens = []
  beam_id = 0
  for prediction in predictions:
    # get the top translation.
    if beam_id == 0:
      for sent_id in range(hparams.infer_batch_size):
        if sent_id >= prediction["predictions"].shape[0]:
          break
        trans, output_length = nmt_utils.get_translation(
            prediction["predictions"],
            sent_id=sent_id,
            tgt_eos=hparams.eos,
            subword_option=hparams.subword_option)
        translations.append(trans)
        output_tokens.append(output_length)
    beam_id += 1
    if beam_id == hparams.beam_width:
      beam_id = 0

  if only_translate:
    trans_file = hparams.translate_file + '.trans.tok'
  else:
    trans_file = os.path.join(
        hparams.output_dir, "newstest2014_out_{}.tok.de".format(
            pred_estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP)))
  trans_dir = os.path.dirname(trans_file)
  if not tf.gfile.Exists(trans_dir):
    tf.gfile.MakeDirs(trans_dir)
  tf.logging.info("Writing to file %s" % trans_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(trans_file,
                                                mode="wb")) as trans_f:
    trans_f.write("")  # Write empty string to ensure file is created.
    for translation in translations:
      trans_f.write((translation + b"\n").decode("utf-8"))

  if only_translate:
    return None, benchmark_hook.get_average_speed_and_latencies(), sum(output_tokens)

  # Evaluation
  output_dir = os.path.join(pred_estimator.model_dir, "eval")
  tf.gfile.MakeDirs(output_dir)
  summary_writer = tf.summary.FileWriter(output_dir)

  ref_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
  # Hardcoded.
  metric = "bleu"
  score = get_sacrebleu(trans_file, hparams.detokenizer_file)

  misc_utils.print_out("bleu is %.5f" % score)
  with tf.Graph().as_default():
    summaries = []
    summaries.append(tf.Summary.Value(tag=metric, simple_value=score))
  tf_summary = tf.Summary(value=list(summaries))
  summary_writer.add_summary(
      tf_summary, pred_estimator.get_variable_value(tf.GraphKeys.GLOBAL_STEP))

  summary_writer.close()
  return score, benchmark_hook.get_average_speed_and_latencies(), sum(output_tokens)


def train_fn(hparams):
  """Train function."""
  model_fn = make_model_fn(hparams)
  input_fn = make_input_fn(hparams, tf.contrib.learn.ModeKeys.TRAIN)

  log_step_count_steps = hparams.log_step_count_steps
  save_checkpoints_steps = hparams.save_checkpoints_steps
  if hparams.use_dist_strategy:
    distribution_strategy = get_distribution_strategy(hparams.num_gpus)
    config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        log_step_count_steps=log_step_count_steps,
        keep_checkpoint_max=None,
        save_checkpoints_steps=save_checkpoints_steps)
  else:
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    if hparams.use_autojit_xla:
      sess_config.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_1)
    if not hparams.use_pintohost_optimizer:
      sess_config.graph_options.rewrite_options.pin_to_host_optimization = (
          rewriter_config_pb2.RewriterConfig.OFF)
    config = tf.estimator.RunConfig(
        log_step_count_steps=log_step_count_steps,
        session_config=sess_config,
        keep_checkpoint_max=None,
        save_checkpoints_steps=save_checkpoints_steps)

  misc_utils.print_out("sess master is %s" % config.master)
  estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=hparams.output_dir, config=config)

  benchmark_hook = BenchmarkHook(hparams.batch_size, hparams.warmup_steps + 5)
  train_hooks = [benchmark_hook]
  if hparams.profile:
    train_hooks.append(tf.train.ProfilerHook(
        output_dir=hparams.output_dir,
        save_steps=hparams.profile_save_steps,
        show_dataflow=True,
        show_memory=True))

  max_steps = hparams.debug_num_train_steps
  estimator.train(
      input_fn=input_fn,
      max_steps=max_steps,
      hooks=train_hooks,
  )

  return benchmark_hook.get_average_speed_and_latencies()


def eval_fn(hparams, ckpt=None, only_translate=False):
  model_fn = make_model_fn(hparams)
  return get_metrics(hparams, model_fn, ckpt, only_translate=only_translate)
