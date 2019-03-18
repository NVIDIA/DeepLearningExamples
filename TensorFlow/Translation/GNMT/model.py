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
"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os

import tensorflow as tf
import numpy as np


from tensorflow.python.framework import function
from tensorflow.python.ops import math_ops

import attention_wrapper
import model_helper
import beam_search_decoder
from utils import iterator_utils
from utils import math_utils
from utils import misc_utils as utils
from utils import vocab_utils

utils.check_tensorflow_version()

__all__ = ["BaseModel"]


def create_attention_mechanism(
    num_units, memory, source_sequence_length, dtype=None):
  """Create attention mechanism based on the attention_option."""
  # Mechanism
  attention_mechanism = attention_wrapper.BahdanauAttention(
      num_units,
      memory,
      memory_sequence_length=tf.to_int64(source_sequence_length),
      normalize=True, dtype=dtype)
  return attention_mechanism


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self, hparams, mode, features, scope=None, extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      features: a dict of input features.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.

    """
    self.hparams = hparams
    # Set params
    self._set_params_initializer(hparams, mode, features, scope, extra_args)

    # Train graph
    res = self.build_graph(hparams, scope=scope)
    self._set_train_or_infer(res, hparams)

  def _set_params_initializer(self,
                              hparams,
                              mode,
                              features,
                              scope,
                              extra_args=None):
    """Set various params for self and initialize."""
    self.mode = mode
    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.features = features
    self.time_major = hparams.time_major

    if hparams.use_char_encode:
      assert (not self.time_major), ("Can't use time major for"
                                     " char-level inputs.")

    self.dtype = tf.float16 if hparams.use_fp16 else tf.float32

    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Set num units
    self.num_units = hparams.num_units
    # Set num layers
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers
    assert self.num_decoder_layers

    # Set num residual layers
    if hasattr(hparams, "num_residual_layers"):  # compatible common_test_utils
      self.num_encoder_residual_layers = hparams.num_residual_layers
      self.num_decoder_residual_layers = hparams.num_residual_layers
    else:
      self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
      self.num_decoder_residual_layers = hparams.num_decoder_residual_layers

    # Batch size
    self.batch_size = tf.size(self.features["source_sequence_length"])

    # Global step
    global_step = tf.train.get_global_step()
    if global_step is not None:
      utils.print_out("global_step already created!")

    self.global_step = tf.train.get_or_create_global_step()
    utils.print_out("model.global_step.name: %s" % self.global_step.name)

    # Initializer
    self.random_seed = hparams.random_seed
    initializer = model_helper.get_initializer(
        hparams.init_op, self.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    self.encoder_emb_lookup_fn = tf.nn.embedding_lookup
    self.init_embeddings(hparams, scope)

  def _set_train_or_infer(self, res, hparams):
    """Set up training."""
    loss = res[1]
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = loss
      self.word_count = tf.reduce_sum(
          self.features["source_sequence_length"]) + tf.reduce_sum(
              self.features["target_sequence_length"])
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = loss
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits = res[0]
      self.infer_loss = loss
      self.sample_id = res[2]

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(
          self.features["target_sequence_length"])

    # Gradients and SGD update operation for training the model.
    # Arrange for the embedding vars to appear at the beginning.
    # Only build bprop if running on GPU and using dist_strategy, in which
    # case learning rate, grads and train_op are created in estimator model
    # function.
    with tf.name_scope("learning_rate"):
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

    if (hparams.use_dist_strategy and
        self.mode == tf.contrib.learn.ModeKeys.TRAIN):
      # Gradients
      params = tf.trainable_variables()
      # Print trainable variables
      utils.print_out("# Trainable variables")
      utils.print_out(
          "Format: <name>, <shape>, <dtype>, <(soft) device placement>")
      for param in params:
        utils.print_out(
            "  %s, %s, %s, %s" % (param.name, str(param.get_shape()),
                                  param.dtype.name, param.op.device))
      utils.print_out("Total params size: %.2f GB" % (4. * np.sum([
          p.get_shape().num_elements()
          for p in params
          if p.shape.is_fully_defined()
      ]) / 2**30))

      # Optimizer
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)
      else:
        raise ValueError("Unknown optimizer type %s" % hparams.optimizer)
      assert opt is not None

      grads_and_vars = opt.compute_gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)
      gradients = [x for (x, _) in grads_and_vars]

      clipped_grads, grad_norm = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.grad_norm = grad_norm
      self.params = params
      self.grads = clipped_grads

      self.update = opt.apply_gradients(
          list(zip(clipped_grads, params)), global_step=self.global_step)
    else:
      self.grad_norm = None
      self.update = None
      self.params = None
      self.grads = None

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))
    if not warmup_scheme:
      return self.learning_rate

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def _get_decay_info(self, hparams):
    """Return decay info based on decay_scheme."""
    if hparams.decay_scheme in [
        "luong5", "luong10", "luong234", "jamesqin1616"
    ]:
      epoch_size, _, _ = iterator_utils.get_effective_epoch_size(hparams)
      num_train_steps = int(hparams.max_train_epochs * epoch_size / hparams.batch_size)
      decay_factor = 0.5
      if hparams.decay_scheme == "luong5":
        start_decay_step = int(num_train_steps / 2)
        decay_times = 5
        remain_steps = num_train_steps - start_decay_step
      elif hparams.decay_scheme == "luong10":
        start_decay_step = int(num_train_steps / 2)
        decay_times = 10
        remain_steps = num_train_steps - start_decay_step
      elif hparams.decay_scheme == "luong234":
        start_decay_step = int(num_train_steps * 2 / 3)
        decay_times = 4
        remain_steps = num_train_steps - start_decay_step
      elif hparams.decay_scheme == "jamesqin1616":
        # dehao@ reported TPU setting max_epoch = 2 and use luong234.
        # They start decay after 2 * 2/3 epochs for 4 times.
        # If keep max_epochs = 8 then decay should start at 8 * 2/(3 * 4) epochs
        # and for (4 *4 = 16) times.
        decay_times = 16
        start_decay_step = int(num_train_steps / 16.)
        remain_steps = num_train_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not hparams.decay_scheme:  # no decay
      start_decay_step = num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    return start_decay_step, decay_steps, decay_factor

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    start_decay_step, decay_steps, decay_factor = self._get_decay_info(hparams)
    utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                    "decay_factor %g" % (hparams.decay_scheme, start_decay_step,
                                         decay_steps, decay_factor))

    return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(  # pylint: disable=g-long-lambda
            self.learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

  def init_embeddings(self, hparams, scope):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=self.num_units,
            tgt_embed_size=self.num_units,
            dtype=self.dtype,
            num_enc_partitions=hparams.num_enc_emb_partitions,
            num_dec_partitions=hparams.num_dec_emb_partitions,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
            use_char_encode=hparams.use_char_encode,
            scope=scope,
        ))

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss_tuple, final_context_state, sample_id),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: loss = the total loss / batch_size.
        final_context_state: the final state of decoder RNN.
        sample_id: sampling indices.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# Creating %s graph ..." % self.mode)

    # Projection
    with tf.variable_scope(scope or "build_network"):
      with tf.variable_scope("decoder/output_projection"):
        self.output_layer = tf.layers.Dense(
            self.tgt_vocab_size, use_bias=False, name="output_projection",
            dtype=self.dtype)

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=self.dtype):
      # Encoder
      if hparams.language_model:  # no encoder for language modeling
        utils.print_out("  language modeling: no encoder")
        self.encoder_outputs = None
        encoder_state = None
      else:
        self.encoder_outputs, encoder_state = self._build_encoder(hparams)

      ## Decoder
      logits, sample_id = (
          self._build_decoder(self.encoder_outputs, encoder_state, hparams))

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss = self._compute_loss(logits, hparams.label_smoothing)
      else:
        loss = tf.constant(0.0)

    return logits, loss, sample_id

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
    """Maximum decoding steps at inference time."""
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      # TODO(thangluong): add decoding_length_factor flag
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(source_sequence_length)
      maximum_iterations = tf.to_int32(
          tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # [batch, time]
        target_input = self.features["target_input"]
        if self.time_major:
          # If using time_major mode, then target_input should be [time, batch]
          # then the decoder_emb_inp would be [time, batch, dim]
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.cast(
            tf.nn.embedding_lookup(self.embedding_decoder, target_input),
            self.dtype)

        if not hparams.use_fused_lstm_dec:
          cell, decoder_initial_state = self._build_decoder_cell(
              hparams, encoder_outputs, encoder_state,
              self.features["source_sequence_length"])

          if hparams.use_dynamic_rnn:
            final_rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell,
                decoder_emb_inp,
                sequence_length=self.features["target_sequence_length"],
                initial_state=decoder_initial_state,
                dtype=self.dtype,
                scope=decoder_scope,
                parallel_iterations=hparams.parallel_iterations,
                time_major=self.time_major)
          else:
            final_rnn_outputs, _ = tf.contrib.recurrent.functional_rnn(
                cell,
                decoder_emb_inp,
                sequence_length=tf.to_int32(
                    self.features["target_sequence_length"]),
                initial_state=decoder_initial_state,
                dtype=self.dtype,
                scope=decoder_scope,
                time_major=self.time_major,
                use_tpu=False)
        else:
          if hparams.pass_hidden_state:
            decoder_initial_state = encoder_state
          else:
            decoder_initial_state = tuple((tf.nn.rnn_cell.LSTMStateTuple(
              tf.zeros_like(s[0]), tf.zeros_like(s[1])) for s in encoder_state))
          final_rnn_outputs = self._build_decoder_fused_for_training(
              encoder_outputs, decoder_initial_state, decoder_emb_inp, self.hparams)

        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        logits = self.output_layer(final_rnn_outputs)
        sample_id = None
      ## Inference
      else:
        cell, decoder_initial_state = self._build_decoder_cell(
            hparams, encoder_outputs, encoder_state,
            self.features["source_sequence_length"])

        assert hparams.infer_mode == "beam_search"
        _, tgt_vocab_table = vocab_utils.create_vocab_tables(
            hparams.src_vocab_file, hparams.tgt_vocab_file, hparams.share_vocab)
        tgt_sos_id = tf.cast(
            tgt_vocab_table.lookup(tf.constant(hparams.sos)), tf.int32)
        tgt_eos_id = tf.cast(
            tgt_vocab_table.lookup(tf.constant(hparams.eos)), tf.int32)
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        coverage_penalty_weight = hparams.coverage_penalty_weight

        my_decoder = beam_search_decoder.BeamSearchDecoder(
            cell=cell,
            embedding=self.embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
            beam_width=beam_width,
            output_layer=self.output_layer,
            length_penalty_weight=length_penalty_weight,
            coverage_penalty_weight=coverage_penalty_weight)

        # maximum_iteration: The maximum decoding steps.
        maximum_iterations = self._get_infer_maximum_iterations(
            hparams, self.features["source_sequence_length"])

        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        logits = tf.no_op()
        sample_id = outputs.predicted_ids

    return logits, sample_id

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder and the initial state of
      the decoder RNN.
    """
    pass

  def _softmax_cross_entropy_loss(self, logits, labels, label_smoothing):
    """Compute softmax loss or sampled softmax loss."""
    use_defun = os.environ["use_defun"] == "true"
    use_xla = os.environ["use_xla"] == "true"

    # @function.Defun(noinline=True, compiled=use_xla)
    def ComputePositiveCrossent(labels, logits):
      crossent = math_utils.sparse_softmax_crossent_with_logits(
          labels=labels, logits=logits)
      return crossent
    crossent = ComputePositiveCrossent(labels, logits)
    assert crossent.dtype == tf.float32

    def _safe_shape_div(x, y):
      """Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`."""
      return x // tf.maximum(y, 1)

    @function.Defun(tf.float32, tf.float32, compiled=use_xla)
    def ReduceSumGrad(x, grad):
      """docstring."""
      input_shape = tf.shape(x)
      # TODO(apassos) remove this once device placement for eager ops makes more
      # sense.
      with tf.colocate_with(input_shape):
        output_shape_kept_dims = math_ops.reduced_shape(input_shape, -1)
        tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
      grad = tf.reshape(grad, output_shape_kept_dims)
      return tf.tile(grad, tile_scaling)

    def ReduceSum(x):
      """docstring."""
      return tf.reduce_sum(x, axis=-1)
    if use_defun:
      ReduceSum = function.Defun(
          tf.float32,
          compiled=use_xla,
          noinline=True,
          grad_func=ReduceSumGrad)(ReduceSum)

    if abs(label_smoothing) > 1e-3:
      # pylint:disable=invalid-name
      def ComputeNegativeCrossentFwd(logits):
        """docstring."""
        # [time, batch, dim]
        # [time, batch]
        max_logits = tf.reduce_max(logits, axis=-1)
        # [time, batch, dim]
        shifted_logits = logits - tf.expand_dims(max_logits, axis=-1)
        # Always compute loss in fp32
        shifted_logits = tf.to_float(shifted_logits)
        # [time, batch]
        log_sum_exp = tf.log(ReduceSum(tf.exp(shifted_logits)))
        # [time, batch, dim] - [time, batch, 1] --> reduce_sum(-1) -->
        # [time, batch]
        neg_crossent = ReduceSum(
            shifted_logits - tf.expand_dims(log_sum_exp, axis=-1))
        return neg_crossent

      def ComputeNegativeCrossent(logits):
        return ComputeNegativeCrossentFwd(logits)

      if use_defun:
        ComputeNegativeCrossent = function.Defun(
            compiled=use_xla)(ComputeNegativeCrossent)

      neg_crossent = ComputeNegativeCrossent(logits)
      neg_crossent = tf.to_float(neg_crossent)
      num_labels = logits.shape[-1].value
      crossent = (1.0 - label_smoothing) * crossent - (
          label_smoothing / tf.to_float(num_labels) * neg_crossent)
      # pylint:enable=invalid-name

    return crossent

  def _compute_loss(self, logits, label_smoothing):
    """Compute optimization loss."""
    target_output = self.features["target_output"]
    if self.time_major:
      target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)
    self.batch_seq_len = max_time

    crossent = self._softmax_cross_entropy_loss(
        logits, target_output, label_smoothing)
    assert crossent.dtype == tf.float32

    target_weights = tf.sequence_mask(
        self.features["target_sequence_length"], max_time, dtype=crossent.dtype)
    if self.time_major:
      # [time, batch] if time_major, since the crossent is [time, batch] in this
      # case.
      target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(
        self.batch_size)

    return loss

  def build_encoder_states(self, include_embeddings=False):
    """Stack encoder states and return tensor [batch, length, layer, size]."""
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    if include_embeddings:
      stack_state_list = tf.stack(
          [self.encoder_emb_inp] + self.encoder_state_list, 2)
    else:
      stack_state_list = tf.stack(self.encoder_state_list, 2)

    # transform from [length, batch, ...] -> [batch, length, ...]
    if self.time_major:
      stack_state_list = tf.transpose(stack_state_list, [1, 0, 2, 3])

    return stack_state_list
