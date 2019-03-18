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
"""For loading data into NMT models."""
from __future__ import print_function

import os

import tensorflow as tf

from utils import vocab_utils


def get_effective_epoch_size(hparams, train=True):
  """Get training epoch size after filtering."""
  if train:
    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    src_max_len = hparams.src_max_len
    tgt_max_len = hparams.tgt_max_len
  else:
    src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    src_max_len = hparams.src_max_len_infer
    tgt_max_len = None

  if src_max_len is None:
    src_max_len = float('inf')
  if tgt_max_len is None:
    tgt_max_len = float('inf')

  srcf = tf.gfile.GFile(src_file, "r")
  tgtf = tf.gfile.GFile(tgt_file, "r")

  epoch_size = 0
  src_tokens = 0
  tgt_tokens = 0
  for srcline, tgtline in zip(srcf, tgtf):
    len_srcline = len(srcline.split())
    len_tgtline = len(tgtline.split())
    if (
        len_srcline < src_max_len and
        len_tgtline < tgt_max_len):
      epoch_size += 1
      src_tokens += len_srcline
      tgt_tokens += len_tgtline
  srcf.close()
  tgtf.close()
  return epoch_size, src_tokens, tgt_tokens


# pylint: disable=g-long-lambda,line-too-long
def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True,
                 use_char_encode=False,
                 num_repeat=1,
                 filter_oversized_sequences=False):
  """Function that returns input dataset."""
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  if use_char_encode:
    src_eos_id = vocab_utils.EOS_CHAR_ID
  else:
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)

  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed,
      reshuffle_each_iteration).repeat(num_repeat)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  # Filter oversized input sequences.
  if filter_oversized_sequences:
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) < src_max_len,
                                        tf.size(tgt) < tgt_max_len))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  if use_char_encode:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.reshape(vocab_utils.tokens_to_bytes(src), [-1]),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls)
  else:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  if use_char_encode:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out,
            tf.to_int32(tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN),
            tf.size(tgt_in)),
        num_parallel_calls=num_parallel_calls)
  else:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  use_xla_compile = os.environ["xla_compile"] == "true"
  force_inputs_padding = os.environ["force_inputs_padding"] == "true"
  use_static_input_shape = use_xla_compile or force_inputs_padding

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape(
                [src_max_len if use_static_input_shape else None]),  # src
            tf.TensorShape(
                [tgt_max_len if use_static_input_shape else None]),  # tgt_input
            tf.TensorShape([tgt_max_len if use_static_input_shape else None
                           ]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0),
        drop_remainder=True)

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      """Calculate bucket_width by maximum source sequence length."""
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
  else:
    batched_dataset = batching_func(src_tgt_dataset)


# Make_one_shot_iterator is not applicable here since we have lookup table.
# Instead return a tf.data.dataset and let TpuEstimator to initialize and make
# iterator out of it.
  batched_dataset = batched_dataset.map(
      lambda src, tgt_in, tgt_out, source_size, tgt_in_size: (
          {"source": src,
           "target_input": tgt_in,
           "target_output": tgt_out,
           "source_sequence_length": source_size,
           "target_sequence_length": tgt_in_size}))
  return batched_dataset


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       src_max_len=None,
                       use_char_encode=False):
  """Get dataset for inference."""
  if use_char_encode:
    src_eos_id = vocab_utils.EOS_CHAR_ID
  else:
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])

  if use_char_encode:
    # Convert the word strings to character ids
    src_dataset = src_dataset.map(
        lambda src: tf.reshape(vocab_utils.tokens_to_bytes(src), [-1]))
  else:
    # Convert the word strings to ids
    src_dataset = src_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

  # Add in the word counts.
  if use_char_encode:
    src_dataset = src_dataset.map(
        lambda src: (src,
                     tf.to_int32(
                         tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN)))
  else:
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0))  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_dataset = batched_dataset.map(
      lambda src_ids, src_seq_len: (
          {"source": src_ids,
           "source_sequence_length": src_seq_len}))
  return batched_dataset
