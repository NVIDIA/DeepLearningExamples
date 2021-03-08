from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from functools import partial

from collections import Counter, OrderedDict
import pickle
import json
import multiprocessing as mp

import numpy as np

from absl import flags
import tensorflow as tf
from vocabulary import Vocab

from tensorflow.gfile import Exists as exists
from tensorflow.gfile import MakeDirs as makedirs
from tensorflow.gfile import Glob as glob


def _preprocess(shard, train, vocab, save_dir, cutoffs, bin_sizes, bsz, tgt_len,
                num_core_per_host, num_shuffle):
  file_names = []
  num_batch = 0

  path = train[shard]
  data_shard = vocab.encode_file(path, ordered=False, add_double_eos=True)

  for shuffle in range(num_shuffle):
    basename = "train-{:03d}-{:02d}".format(shard, shuffle)
    print("Processing shard {} shuffle {}".format(shard, shuffle))

    np.random.shuffle(data_shard)
    file_name, num_batch_shuffle = create_ordered_tfrecords(
        save_dir, basename, np.concatenate(data_shard), bsz, tgt_len,
        num_core_per_host, cutoffs, bin_sizes)
    file_names.append(file_name)
    num_batch += num_batch_shuffle

  return file_names, num_batch


class Corpus(object):
  def __init__(self, path, dataset, *args, **kwargs):
    self.dataset = dataset
    self.vocab = Vocab(*args, **kwargs)

    if self.dataset in ["ptb", "wt2", "enwik8", "text8"]:
      self.vocab.count_file(os.path.join(path, "train.txt"))
      self.vocab.count_file(os.path.join(path, "valid.txt"))
      self.vocab.count_file(os.path.join(path, "test.txt"))
    elif self.dataset == "wt103":
      self.vocab.count_file(os.path.join(path, "train.txt"))
    elif self.dataset == "lm1b":
      train_path_pattern = os.path.join(
          path, "1-billion-word-language-modeling-benchmark-r13output",
          "training-monolingual.tokenized.shuffled", "news.en-*")
      train_paths = glob(train_path_pattern)

      # the vocab will load from file when build_vocab() is called
      # for train_path in sorted(train_paths):
      #   self.vocab.count_file(train_path, verbose=True)

    self.vocab.build_vocab()

    if self.dataset in ["ptb", "wt2", "wt103"]:
      self.train = self.vocab.encode_file(
          os.path.join(path, "train.txt"), ordered=True)
      self.valid = self.vocab.encode_file(
          os.path.join(path, "valid.txt"), ordered=True)
      self.test  = self.vocab.encode_file(
          os.path.join(path, "test.txt"), ordered=True)
    elif self.dataset in ["enwik8", "text8"]:
      self.train = self.vocab.encode_file(
          os.path.join(path, "train.txt"), ordered=True, add_eos=False)
      self.valid = self.vocab.encode_file(
          os.path.join(path, "valid.txt"), ordered=True, add_eos=False)
      self.test  = self.vocab.encode_file(
          os.path.join(path, "test.txt"), ordered=True, add_eos=False)
    elif self.dataset == "lm1b":
      self.train = train_paths
      valid_path = os.path.join(path, "valid.txt")
      test_path = valid_path
      self.valid = self.vocab.encode_file(
          valid_path, ordered=True, add_double_eos=True)
      self.test  = self.vocab.encode_file(
          test_path, ordered=True, add_double_eos=True)

    if self.dataset == "wt103":
      self.cutoffs = [0, 19997, 39997, 199997] + [len(self.vocab)]
    elif self.dataset == "lm1b":
      self.cutoffs = [0, 59997, 99997, 639997] + [len(self.vocab)]
    else:
      self.cutoffs = []


  def convert_to_tfrecords(self, split, save_dir, bsz, tgt_len,
                           num_core_per_host, **kwargs):
    FLAGS = kwargs.get('FLAGS')

    file_names = []

    record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
        split, bsz, tgt_len)

    record_info_path = os.path.join(save_dir, record_name)

    if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
      data = getattr(self, split)
      bin_sizes = get_bin_sizes(
          data, bsz // num_core_per_host, tgt_len, self.cutoffs)
      file_name, num_batch = create_ordered_tfrecords(
          save_dir, split, data, bsz, tgt_len, num_core_per_host,
          self.cutoffs, bin_sizes,
          num_passes=FLAGS.num_passes if split == 'train' else 1)
      file_names.append(file_name)
    elif self.dataset == "lm1b":
      bin_sizes = get_bin_sizes(
          self.valid, bsz // num_core_per_host, tgt_len, self.cutoffs)
      if split == "train":
        np.random.seed(123456)
        num_batch = 0

        if FLAGS.num_procs > 1:
          _preprocess_wrapper = partial(_preprocess,
              train=self.train, vocab=self.vocab, save_dir=save_dir,
              cutoffs=self.cutoffs, bin_sizes=bin_sizes, bsz=bsz,
              tgt_len=tgt_len, num_core_per_host=num_core_per_host,
              num_shuffle=FLAGS.num_shuffle)

          pool = mp.Pool(processes=FLAGS.num_procs)
          results = pool.map(_preprocess_wrapper, range(len(self.train)))
          for res in results:
            file_names.extend(res[0])
            num_batch += res[1]
        else:
          for shard, path in enumerate(self.train):
            data_shard = self.vocab.encode_file(path, ordered=False,
                                                add_double_eos=True)

            num_shuffle = FLAGS.num_shuffle

            for shuffle in range(num_shuffle):
              print("Processing shard {} shuffle {}".format(shard, shuffle))
              basename = "train-{:03d}-{:02d}".format(shard, shuffle)
              np.random.shuffle(data_shard)
              file_name, num_batch_ = create_ordered_tfrecords(
                  save_dir, basename, np.concatenate(data_shard), bsz, tgt_len,
                  num_core_per_host,
                  self.cutoffs, bin_sizes)
              file_names.append(file_name)
              num_batch += num_batch_

      else:
        file_name, num_batch = create_ordered_tfrecords(
            save_dir, split, getattr(self, split), bsz, tgt_len,
            num_core_per_host,
            self.cutoffs, bin_sizes)
        file_names.append(file_name)

    with open(record_info_path, "w") as fp:
      record_info = {
        "filenames": file_names,
        "bin_sizes": bin_sizes,
        "num_batch": num_batch
      }
      json.dump(record_info, fp)


def get_bin_sizes(data, batch_size, tgt_len, cutoffs, std_mult=[2.5, 2.5, 2.5]):
  """
    Note: the `batch_size` here should be per-core batch size
  """
  bin_sizes = []

  def _nearest_to_eight(x):
    y = x - x % 8
    return y + 8 if x % 8 >= 4 else max(8, y)

  if cutoffs:
    num_batch = len(data) // batch_size // tgt_len

    data = data[:batch_size * num_batch * tgt_len]
    data = data.reshape(batch_size, num_batch, tgt_len)

    tot = batch_size * tgt_len
    for b, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):
      mask = (data >= left) * (data < right)
      percents = mask.astype(np.float64).sum(2).sum(0) / tot
      mean = np.mean(percents)
      std = np.std(percents)

      bin_size = int(math.ceil(tgt_len * batch_size * (mean + std_mult[b] * std)))
      bin_size = _nearest_to_eight(bin_size)
      bin_sizes.append(bin_size)

  return bin_sizes


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def batchify(data, batch_size, num_passes):
  """
    if num_passes > 1

    Here, we use multiple randomly shifted copies.
  """
  if num_passes > 1:
    data_len = len(data)
    double_data = np.concatenate([data, data])
    data_list = []
    for i in range(num_passes):
      start = np.random.randint(0, data_len)
      data_list.append(double_data[start:start+data_len])
    data = np.concatenate(data_list)

  num_step = len(data) // batch_size
  data = data[:batch_size * num_step]
  data = data.reshape(batch_size, num_step)

  return data


def create_ordered_tfrecords(save_dir, basename, data, batch_size, tgt_len,
                             num_core_per_host, cutoffs=[], bin_sizes=[],
                             num_passes=1):

  file_name = "{}.bsz-{}.tlen-{}.tfrecords".format(
      basename, batch_size, tgt_len)

  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.python_io.TFRecordWriter(save_path)

  batched_data = batchify(data, batch_size, num_passes)

  num_batch = 0
  for t in range(0, batched_data.shape[1] - 1, tgt_len):
    cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
    if num_batch % 500 == 0:
      print("  processing batch {}".format(num_batch))
    for idx in range(batch_size):
      inputs = batched_data[idx, t:t + cur_tgt_len]
      labels = batched_data[idx, t + 1:t + cur_tgt_len + 1]

      # features dict
      feature = {
          "inputs": _int64_feature(inputs),
          "labels": _int64_feature(labels),
      }

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      record_writer.write(example.SerializeToString())

    num_batch += 1

  record_writer.close()
  print("Done writing {}. batches: {}".format(file_name, num_batch))

  return file_name, num_batch


def get_lm_corpus(data_dir, dataset):
  fn = os.path.join(data_dir, "cache.pkl")

  if exists(fn):
    print("Loading cached dataset...")
    with open(fn, "rb") as fp:
      corpus = pickle.load(fp)
  else:
    print("Producing dataset...")
    kwargs = {}
    if dataset in ["wt103", "wt2"]:
      kwargs["special"] = ["<eos>"]
      kwargs["lower_case"] = False
    elif dataset == "ptb":
      kwargs["special"] = ["<eos>"]
      kwargs["lower_case"] = True
    elif dataset == "lm1b":
      kwargs["special"] = []
      kwargs["lower_case"] = False
      kwargs["vocab_file"] = os.path.join(data_dir, "1b_word_vocab.txt")
    elif dataset in ["enwik8", "text8"]:
      pass

    corpus = Corpus(data_dir, dataset, **kwargs)

    print("Saving dataset...")
    with open(fn, "wb") as fp:
      pickle.dump(corpus, fp, protocol=2)

    corpus_info = {
      "vocab_size" : len(corpus.vocab),
      "cutoffs" : corpus.cutoffs,
      "dataset" : corpus.dataset
    }
    with open(os.path.join(data_dir, "corpus-info.json"), "w") as fp:
      json.dump(corpus_info, fp)

  return corpus


def main(unused_argv):
  del unused_argv  # Unused

  corpus = get_lm_corpus(FLAGS.data_dir, FLAGS.dataset)

  save_dir = os.path.join(FLAGS.data_dir, "tfrecords")
  if not exists(save_dir):
    makedirs(save_dir)

  # test mode
  if FLAGS.eval_batch_size > 0:
    corpus.convert_to_tfrecords("test", save_dir, FLAGS.eval_batch_size,
                                FLAGS.tgt_len, FLAGS.num_core_per_host,
                                FLAGS=FLAGS)
    return

  for split, batch_size in zip(
      ["train", "valid"],
      [FLAGS.train_batch_size // FLAGS.batch_chunk, FLAGS.valid_batch_size]):

    if batch_size <= 0: continue
    print("Converting {} set...".format(split))
    corpus.convert_to_tfrecords(split, save_dir, batch_size, FLAGS.tgt_len,
                                FLAGS.num_core_per_host, FLAGS=FLAGS)


def load_record_info(record_info_dir, split, per_host_bsz, tgt_len,
                     num_core_per_host):
  record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
      split, per_host_bsz, tgt_len)

  record_info_path = os.path.join(record_info_dir, record_name)
  with open(record_info_path, "r") as fp:
    record_info = json.load(fp)

  return record_info

def get_input_fn(record_info_dir, split, per_host_bsz, tgt_len,
                 num_core_per_host, num_hosts=1):
  """Creates input function."""
  record_info = load_record_info(record_info_dir, split, per_host_bsz, tgt_len,
                                 num_core_per_host)

  file_names = record_info["filenames"]
  bin_sizes = record_info["bin_sizes"]
  num_batch = record_info["num_batch"]

  tf.logging.info("[{}] File names {}".format(split, file_names))

  def input_fn(params):
    # per-core batch size
    per_core_bsz = params["batch_size"] // num_core_per_host

    # data_dir could be a remote path, e.g., a google storage url
    data_dir = params["data_dir"]

    def parser(record):
      # preprocess "inp_perm" and "tgt_perm"
      def _process_perm_feature(example, prefix):
        for b in range(len(bin_sizes)):
          cnt = example.pop("{}_cnt_{}".format(prefix, b))[0]
          tup = example.pop("{}_tup_{}".format(prefix, b))

          tup = tf.reshape(
              tf.sparse_tensor_to_dense(tup),
              shape=[cnt, 2])

          # tf.float32
          perm = tf.sparse_to_dense(
              sparse_indices=tup,
              output_shape=[tgt_len, bin_sizes[b]],
              sparse_values=1.0,
              default_value=0.0)

          example["{}_perm_{}".format(prefix, b)] = perm

      # whether allow the last batch with a potentially shorter length
      record_spec = {
          "inputs": tf.VarLenFeature(tf.int64),
          "labels": tf.VarLenFeature(tf.int64),
      }

      # retrieve serialized example
      example = tf.parse_single_example(
          serialized=record,
          features=record_spec)

      # cast int64 into int32
      # cast sparse to dense
      for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
          val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
          val = tf.to_int32(val)
        example[key] = val

      return example["inputs"], example["labels"]

    file_paths = []
    for file_name in file_names:
      file_path = os.path.join(data_dir, file_name)
      file_paths.append(file_path)

    if split == "train":
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      if len(file_paths) > 1:
        dataset = dataset.shuffle(len(file_paths)).repeat()
        dataset = tf.data.TFRecordDataset(dataset)
      elif num_hosts > 1:
        host_id = params["context"].current_host
        # drop the remaining batches
        num_batch_per_host = num_batch // num_hosts

        my_start_sample_id = (host_id * num_batch_per_host * num_core_per_host *
                              per_core_bsz)
        my_sample_num = num_batch_per_host * num_core_per_host * per_core_bsz
        dataset = tf.data.TFRecordDataset(dataset).skip(
            my_start_sample_id).take(my_sample_num)
      else:
        dataset = tf.data.TFRecordDataset(dataset)

      if num_core_per_host > 1:
        import horovod.tensorflow as hvd
        dataset = dataset.shard(hvd.size(), hvd.rank())
      dataset = dataset.map(parser).cache().repeat()
      dataset = dataset.batch(per_core_bsz, drop_remainder=True)
      dataset = dataset.prefetch(num_core_per_host * per_core_bsz)
    else:
      # do not shuffle, repeat or cache in evaluation
      dataset = tf.data.Dataset.from_tensor_slices(file_paths)
      dataset = tf.data.TFRecordDataset(dataset)
      dataset = dataset.map(parser)
      dataset = dataset.batch(per_core_bsz, drop_remainder=True)

    return dataset

  if split == "train" and num_hosts > 1:
    record_info["num_batch"] = num_batch // num_hosts

  return input_fn, record_info

def get_corpus_info(corpus_info_path):
  with open(corpus_info_path, "r") as fp:
    corpus_info = json.load(fp)
  return corpus_info

if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_string("data_dir", None,
        help="Location of the data corpus")
  flags.DEFINE_enum("dataset", "wt103",
        ["ptb", "wt2", "wt103", "lm1b", "enwik8", "text8"],
        help="Dataset name.")
  flags.DEFINE_integer("train_batch_size", 256,
        help="train batch size each host")
  flags.DEFINE_integer("valid_batch_size", 256,
        help="valid batch size each host")
  flags.DEFINE_integer("eval_batch_size", 16,
        help="If > 0, enter test mode and process test set only."
             "Otherwise, process train and dev sets only.")
  flags.DEFINE_integer("tgt_len", 70,
        help="number of tokens to predict")
  flags.DEFINE_integer("max_batch", -1,
        help="run in debug mode")
  flags.DEFINE_integer("num_core_per_host", 8,
        help="number of GPUs per host")
  flags.DEFINE_bool("debug", default=False,
        help="Process only the first batch without shuffle for lm1b.")
  flags.DEFINE_integer("num_procs", 1,
        help="number of processes")
  flags.DEFINE_integer("num_passes", 10,
        help="number of passes")
  flags.DEFINE_integer("num_shuffle", 4,
        help="number of shuffles for lm1b")
  flags.DEFINE_integer("batch_chunk", 1,
        help="number of accumulation steps")

  tf.app.run(main)
