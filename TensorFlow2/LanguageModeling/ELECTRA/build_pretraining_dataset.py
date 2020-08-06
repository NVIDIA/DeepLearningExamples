# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Writes out text data as tfrecords that ELECTRA can be pre-trained on."""

import argparse
import multiprocessing
import os
import random
import time
import tensorflow.compat.v1 as tf

import utils
from tokenization import ElectraTokenizer



def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


class ExampleBuilder(object):
  """Given a stream of input text, creates pretraining examples."""

  def __init__(self, tokenizer, max_length):
    self._tokenizer = tokenizer
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length

  def add_line(self, line):
    """Adds a line of text to the current example being built."""
    line = line.strip().replace("\n", " ")
    if (not line) and self._current_length != 0:  # empty lines separate docs
      return self._create_example()
    bert_tokens = self._tokenizer.tokenize(line)
    bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
    self._current_sentences.append(bert_tokids)
    self._current_length += len(bert_tokids)
    if self._current_length >= self._target_length:
      return self._create_example()
    return None

  def _create_example(self):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      first_segment_target_length = (self._target_length - 3) // 2

    first_segment = []
    second_segment = []
    for sentence in self._current_sentences:
      # the sentence goes to the first segment if (1) the first segment is
      # empty, (2) the sentence doesn't put the first segment over length or
      # (3) 50% of the time when it does put the first segment over length
      if (len(first_segment) == 0 or
          len(first_segment) + len(sentence) < first_segment_target_length or
          (len(second_segment) == 0 and
           len(first_segment) < first_segment_target_length and
           random.random() < 0.5)):
        first_segment += sentence
      else:
        second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:self._max_length - 2]
    second_segment = second_segment[:max(0, self._max_length -
                                         len(first_segment) - 3)]

    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0
    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length)
    else:
      self._target_length = self._max_length

    return self._make_tf_example(first_segment, second_segment)

  def _make_tf_example(self, first_segment, second_segment):
    """Converts two "segments" of text into a tf.train.Example."""
    vocab = self._tokenizer.vocab
    input_ids = [vocab["[CLS]"]] + first_segment + [vocab["[SEP]"]]
    segment_ids = [0] * len(input_ids)
    if second_segment:
      input_ids += second_segment + [vocab["[SEP]"]]
      segment_ids += [1] * (len(second_segment) + 1)
    input_mask = [1] * len(input_ids)
    input_ids += [0] * (self._max_length - len(input_ids))
    input_mask += [0] * (self._max_length - len(input_mask))
    segment_ids += [0] * (self._max_length - len(segment_ids))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "input_ids": create_int_feature(input_ids),
        "input_mask": create_int_feature(input_mask),
        "segment_ids": create_int_feature(segment_ids)
    }))
    return tf_example


class ExampleWriter(object):
  """Writes pre-training examples to disk."""

  def __init__(self, job_id, vocab_file, output_dir, max_seq_length,
               num_jobs, blanks_separate_docs, do_lower_case,
               num_out_files=1000):
    self._blanks_separate_docs = blanks_separate_docs
    tokenizer = ElectraTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)
    self._example_builder = ExampleBuilder(tokenizer, max_seq_length)
    self._writers = []
    for i in range(num_out_files):
      if i % num_jobs == job_id:
        output_fname = os.path.join(
            output_dir, "pretrain_data.tfrecord-{:}-of-{:}".format(
                i, num_out_files))
        self._writers.append(tf.io.TFRecordWriter(output_fname))
    self.n_written = 0

  def write_examples(self, input_file):
    """Writes out examples from the provided input file."""
    with tf.io.gfile.GFile(input_file) as f:
      for line in f:
        line = line.strip()
        if line or self._blanks_separate_docs:
          example = self._example_builder.add_line(line)
          if example:
            self._writers[self.n_written % len(self._writers)].write(
                example.SerializeToString())
            self.n_written += 1
      example = self._example_builder.add_line("")
      if example:
        self._writers[self.n_written % len(self._writers)].write(
            example.SerializeToString())
        self.n_written += 1

  def finish(self):
    for writer in self._writers:
      writer.close()


def write_examples(job_id, args):
  """A single process creating and writing out pre-processed examples."""

  def log(*args):
    msg = " ".join(map(str, args))
    print("Job {}:".format(job_id), msg)

  log("Creating example writer")
  example_writer = ExampleWriter(
      job_id=job_id,
      vocab_file=args.vocab_file,
      output_dir=args.output_dir,
      max_seq_length=args.max_seq_length,
      num_jobs=args.num_processes,
      blanks_separate_docs=args.blanks_separate_docs,
      do_lower_case=args.do_lower_case,
      num_out_files=args.num_out_files,
  )
  log("Writing tf examples")
  fnames = sorted(tf.io.gfile.listdir(args.corpus_dir))
  fnames = [f for (i, f) in enumerate(fnames)
            if i % args.num_processes == job_id]
  random.shuffle(fnames)
  start_time = time.time()
  for file_no, fname in enumerate(fnames):
    if file_no > 0:
      elapsed = time.time() - start_time
      log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, "
          "{:} examples written".format(
              file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
              int((len(fnames) - file_no) / (file_no / elapsed)),
              example_writer.n_written))
    example_writer.write_examples(os.path.join(args.corpus_dir, fname))
  example_writer.finish()
  log("Done!")

# python build_pretraining_dataset --corpus-dir
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-dir", required=True,
                      help="Location of pre-training text files.")
  parser.add_argument("--vocab-file", required=True,
                      help="Location of vocabulary file.")
  parser.add_argument("--output-dir", required=True,
                      help="Where to write out the tfrecords.")
  parser.add_argument("--max-seq-length", default=128, type=int,
                      help="Number of tokens per example.")
  parser.add_argument("--num-processes", default=1, type=int,
                      help="Parallelize across multiple processes.")
  parser.add_argument("--blanks-separate-docs", default=True, type=bool,
                      help="Whether blank lines indicate document boundaries.")
  parser.add_argument("--do-lower-case", dest='do_lower_case',
                      action='store_true', help="Lower case input text.")
  parser.add_argument("--no-lower-case", dest='do_lower_case',
                      action='store_false', help="Don't lower case input text.")
  parser.add_argument("--num-out-files", default=1000, type=int,
                      help="Number of output files.")
  parser.set_defaults(do_lower_case=True)
  args = parser.parse_args()

  utils.rmkdir(args.output_dir)
  if args.num_processes == 1:
    write_examples(0, args)
  else:
    jobs = []
    for i in range(args.num_processes):
      job = multiprocessing.Process(target=write_examples, args=(i, args))
      jobs.append(job)
      job.start()
    for job in jobs:
      job.join()


if __name__ == "__main__":
  main()