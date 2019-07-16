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

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from utils import misc_utils as utils

# word level special token
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

# char ids 0-255 come from utf-8 encoding bytes
# assign 256-300 to special chars
BOS_CHAR_ID = 256  # <begin sentence>
EOS_CHAR_ID = 257  # <end sentence>
BOW_CHAR_ID = 258  # <begin word>
EOW_CHAR_ID = 259  # <end word>
PAD_CHAR_ID = 260  # <padding>

DEFAULT_CHAR_MAXLEN = 50  # max number of chars for each word.


def _string_to_bytes(text, max_length):
  """Given string and length, convert to byte seq of at most max_length.

  This process mimics docqa/elmo's preprocessing:
  https://github.com/allenai/document-qa/blob/master/docqa/elmo/data.py

  Note that we make use of BOS_CHAR_ID and EOS_CHAR_ID in iterator_utils.py &
  our usage differs from docqa/elmo.

  Args:
    text: tf.string tensor of shape []
    max_length: max number of chars for each word.

  Returns:
    A tf.int32 tensor of the byte encoded text.
  """
  byte_ids = tf.to_int32(tf.decode_raw(text, tf.uint8))
  byte_ids = byte_ids[:max_length - 2]
  padding = tf.fill([max_length - tf.shape(byte_ids)[0] - 2], PAD_CHAR_ID)
  byte_ids = tf.concat(
      [[BOW_CHAR_ID], byte_ids, [EOW_CHAR_ID], padding], axis=0)
  tf.logging.info(byte_ids)

  byte_ids = tf.reshape(byte_ids, [max_length])
  tf.logging.info(byte_ids.get_shape().as_list())
  return byte_ids + 1


def tokens_to_bytes(tokens):
  """Given a sequence of strings, map to sequence of bytes.

  Args:
    tokens: A tf.string tensor

  Returns:
    A tensor of shape words.shape + [bytes_per_word] containing byte versions
    of each word.
  """
  bytes_per_word = DEFAULT_CHAR_MAXLEN
  with tf.device("/cpu:0"):
    tf.assert_rank(tokens, 1)
    shape = tf.shape(tokens)
    tf.logging.info(tokens)
    tokens_flat = tf.reshape(tokens, [-1])
    as_bytes_flat = tf.map_fn(
        fn=lambda x: _string_to_bytes(x, max_length=bytes_per_word),
        elems=tokens_flat,
        dtype=tf.int32,
        back_prop=False)
    tf.logging.info(as_bytes_flat)
    as_bytes = tf.reshape(as_bytes_flat, [shape[0], bytes_per_word])
  return as_bytes


def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size


def check_vocab(vocab_file, output_dir, check_special_token=True, sos=None,
                eos=None, unk=None, pad_vocab=False):
  """Check if vocab_file doesn't exist, create from corpus_file."""
  if tf.gfile.Exists(vocab_file):
    utils.print_out("# Vocab file %s exists" % vocab_file)
    vocab, vocab_size = load_vocab(vocab_file)
    if check_special_token:
      # Verify if the vocab starts with unk, sos, eos
      # If not, prepend those tokens & generate a new vocab file
      if not unk: unk = UNK
      if not sos: sos = SOS
      if not eos: eos = EOS
      assert len(vocab) >= 3
      if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
        utils.print_out("The first 3 vocab words [%s, %s, %s]"
                        " are not [%s, %s, %s]" %
                        (vocab[0], vocab[1], vocab[2], unk, sos, eos))
        vocab = [unk, sos, eos] + vocab
        vocab_size += 3
        new_vocab_file = os.path.join(output_dir, os.path.basename(vocab_file))
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(new_vocab_file, "wb")) as f:
          for word in vocab:
            f.write("%s\n" % word)
        vocab_file = new_vocab_file
    if pad_vocab == True and vocab_size % 8 != 0:
        new_vocab_file = os.path.join(output_dir, os.path.basename(vocab_file))
        padded_vocab_size = ((vocab_size + 8 - 1)// 8) * 8
        for i in range(0, padded_vocab_size - vocab_size):
            token = "<madeupword" + str(i) + ">"
            vocab.append(token)
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(new_vocab_file, "wb")) as f:
            for word in vocab:
                f.write("%s\n" % word)
            vocab_file = new_vocab_file
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)

  vocab_size = len(vocab)
  return vocab_size, vocab_file


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  src_vocab_table = lookup_ops.index_table_from_file(
      src_vocab_file, default_value=UNK_ID)
  if share_vocab:
    tgt_vocab_table = src_vocab_table
  else:
    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID)
  return src_vocab_table, tgt_vocab_table


def load_embed_txt(embed_file):
  """Load embed_file into a python dictionary.

  Note: the embed_file should be a Glove/word2vec formatted txt file. Assuming
  Here is an exampe assuming embed_size=5:

  the -0.071549 0.093459 0.023738 -0.090339 0.056123
  to 0.57346 0.5417 -0.23477 -0.3624 0.4037
  and 0.20327 0.47348 0.050877 0.002103 0.060547

  For word2vec format, the first line will be: <num_words> <emb_size>.

  Args:
    embed_file: file path to the embedding file.
  Returns:
    a dictionary that maps word to vector, and the size of embedding dimensions.
  """
  emb_dict = dict()
  emb_size = None

  is_first_line = True
  with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, "rb")) as f:
    for line in f:
      tokens = line.rstrip().split(" ")
      if is_first_line:
        is_first_line = False
        if len(tokens) == 2:  # header line
          emb_size = int(tokens[1])
          continue
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      if emb_size:
        if emb_size != len(vec):
          utils.print_out(
              "Ignoring %s since embeding size is inconsistent." % word)
          del emb_dict[word]
      else:
        emb_size = len(vec)
  return emb_dict, emb_size
