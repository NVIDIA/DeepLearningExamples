# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Translate text or files using trained transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import horovod.tensorflow as hvd
from mpi4py import MPI

from data.process_data import _VOCAB_FILE
from model import model_params
import transformer_main
from utils import tokenizer
from utils import distributed_utils
from pynvml import *
from numa import *
import sentencepiece as spm
import time

# report samples/sec during training
class _LogSessionRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, display_every=10):
    self.global_batch_size = global_batch_size
    self.display_every = display_every

  def after_create_session(self, session, coord):
    print('|  Step words/sec')
    self.elapsed_secs = 0.
    self.count = 0

  def before_run(self, run_context):
    self.t0 = time.time()
    return tf.train.SessionRunArgs(fetches=['global_step:0'])

  def after_run(self, run_context, run_values):
    self.elapsed_secs += time.time() - self.t0
    self.count += 1
    global_step = run_values.results[0]
    print_step = global_step + 1 # One-based index for printing.
    if not FLAGS.enable_horovod or hvd.rank() == 0:
      dt = self.elapsed_secs / self.count
      img_per_sec = self.global_batch_size / dt
      print('|%6i %9.1f' %(print_step, img_per_sec))
      self.elapsed_secs = 0.
      self.count = 0

def _get_sorted_inputs(filename):
  """Read and sort lines from the file sorted by decreasing length.

  Args:
    filename: String name of file to read inputs from.
  Returns:
    Sorted list of inputs, and dictionary mapping original index->sorted index
    of each element.
  """
  with tf.gfile.Open(filename) as f:
    records = f.read().split("\n")
    inputs = [record.strip() for record in records]
    if not inputs[-1]:
      inputs.pop()

  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

  sorted_inputs = []
  sorted_keys = {}
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


def _encode_and_add_eos(line, subtokenizer):
  """Encode line with subtokenizer, and add EOS id to the end."""
  return subtokenizer.encode_as_ids(line) + [subtokenizer.eos_id()]


def _trim_and_decode(ids, subtokenizer):
  """Trim EOS and PAD tokens from ids, and decode to return a string."""
  ids = ids.tolist()
  try:
    index = ids.index(subtokenizer.eos_id())
    return subtokenizer.decode_ids(ids[:index])
  except ValueError:  # No EOS found in sequence
    return subtokenizer.decode_ids(ids)


def translate_file(params, estimator, subtokenizer, input_file, output_file=None, print_all_translations=True, report_throughput=False):
  """Translate lines in file, and save to output file if specified.

  Args:
    estimator: tf.Estimator used to generate the translations.
    subtokenizer: Subtokenizer object for encoding and decoding source and
       translated lines.
    input_file: file containing lines to translate
    output_file: file that stores the generated translations.
    print_all_translations: If true, all translations are printed to stdout.

  Raises:
    ValueError: if output file is invalid.
  """

  # Read and sort inputs by length. Keep dictionary (original index-->new index
  # in sorted list) to write translations in the original order.
  sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
  global_batch_size = params.decode_batch_size * hvd.size() if params.enable_horovod else params.decode_batch_size
  num_decode_batches = (len(sorted_inputs) - 1) // global_batch_size + 1
  training_hooks = []
  if report_throughput:
    training_hooks.append(_LogSessionRunHook(global_batch_size, params.display_interval))

  def input_generator():
    """Yield encoded strings from sorted_inputs."""
    for i, line in enumerate(sorted_inputs):
      if params.enable_horovod and i % hvd.size() != hvd.rank():
        continue
      if i % global_batch_size == 0:
        batch_num = (i // global_batch_size) + 1
        print("Decoding batch %d out of %d." % (batch_num, num_decode_batches))

      yield _encode_and_add_eos(line, subtokenizer)

  def input_fn():
    """Created batched dataset of encoded inputs."""
    ds = tf.data.Dataset.from_generator(input_generator, tf.int64, tf.TensorShape([None]))
    ds = ds.padded_batch(params.decode_batch_size, [None])
    return ds

  translations = []
  for i, prediction in enumerate(estimator.predict(input_fn, hooks=training_hooks)):
    translation = _trim_and_decode(prediction["outputs"], subtokenizer)
    translations.append(translation)

    if print_all_translations:
      print("Translating:")
      print("\tInput: %s" % sorted_inputs[i])
      print("\tOutput: %s\n" % translation)
      print("=" * 100)

  if params.enable_horovod:
    comm = MPI.COMM_WORLD
    translation_gather = comm.gather(translations, root=0)
    if hvd.rank() == 0:
      translations = []
      for index in xrange(len(translation_gather[0])):
        for sublist in translation_gather:
          if index < len(sublist):
            translations.append(sublist[index])

  # Write translations in the order they appeared in the original file.
  if output_file is not None:
    if tf.gfile.IsDirectory(output_file):
      raise ValueError("File output is a directory, will not save outputs to file.")
    tf.logging.info("Writing to file %s" % output_file)
    with tf.gfile.Open(output_file, "w") as f:
      for index in xrange(len(sorted_keys)):
        f.write("%s\n" % translations[sorted_keys[index]])


def translate_text(params, estimator, subtokenizer, txt, report_throughput=False):
  """Translate a single string."""
  encoded_txt = _encode_and_add_eos(txt, subtokenizer)

  def input_fn():
    ds = tf.data.Dataset.from_tensors(encoded_txt)
    ds = ds.batch(params.decode_batch_size)
    return ds

  global_batch_size = params.decode_batch_size * hvd.size() if params.enable_horovod else params.decode_batch_size
  training_hooks = []
  if report_throughput:
    training_hooks.append(_LogSessionRunHook(global_batch_size, params.display_interval))

  predictions = estimator.predict(input_fn, hooks=training_hooks)
  translation = next(predictions)["outputs"]
  translation = _trim_and_decode(translation, subtokenizer)
  print("Translation of \"%s\": \"%s\"" % (txt, translation))


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.text is None and FLAGS.file is None:
    tf.logging.warn("Nothing to translate. Make sure to call this script using flags --text or --file.")
    return

  if FLAGS.enable_horovod:
    hvd.init()
    distributed_utils.suppress_output()
  if not FLAGS.sentencepiece:
      subtokenizer = tokenizer.Subtokenizer(os.path.join(FLAGS.data_dir, FLAGS.vocab_file))
  else:
      subtokenizer = spm.SentencePieceProcessor()
      subtokenizer.load('{}.model'.format(os.path.join(FLAGS.data_dir, FLAGS.vocab_file)))

  if FLAGS.params == "base" and not FLAGS.enable_fp16:
      params = model_params.TransformerBaseParams
  elif FLAGS.params == "big" and not FLAGS.enable_fp16:
      params = model_params.TransformerBigParams
  elif FLAGS.params == "base" and FLAGS.enable_fp16:
      params = model_params.TransformerBaseFP16Params
  elif FLAGS.params == "big" and FLAGS.enable_fp16:
      params = model_params.TransformerBigFP16Params
  else:
      raise ValueError("Invalid parameter set defined: %s. Expected 'base' or 'big.'" % FLAGS.params)

  params.eos_id = subtokenizer.eos_id()
  params.enable_horovod = FLAGS.enable_horovod
  if FLAGS.batch_size > 0:
    params.decode_batch_size = FLAGS.batch_size

  config = tf.ConfigProto()
  if FLAGS.enable_horovod:
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      # set CPU affinity mask to what NVML is recommending for optimal performance
      nvmlInit()
      handle = nvmlDeviceGetHandleByIndex(hvd.local_rank())
      cpuSet = nvmlDeviceGetCpuAffinity(handle, 4)
      cpuset = []

      for i in range(cpuSet._length_ * 64):
          word = i // 64
          bit = i % 64

          if cpuSet[word] & (1 << bit) != 0:
              cpuset.append(i)

      set_affinity(os.getpid(), cpuset)
      nvmlShutdown()
  run_config = tf.estimator.RunConfig(
    model_dir=FLAGS.model_dir,
    session_config=config)
  # Set up estimator and params
  estimator = tf.estimator.Estimator(model_fn=transformer_main.model_fn,
    model_dir=FLAGS.model_dir,
    params=params,
    config=run_config)

  if FLAGS.text is not None:
    tf.logging.info("Translating text: %s" % FLAGS.text)
    translate_text(params, estimator, subtokenizer, FLAGS.text, FLAGS.report_throughput)

  if FLAGS.file is not None:
    input_file = os.path.abspath(FLAGS.file)
    tf.logging.info("Translating file: %s" % input_file)
    if not tf.gfile.Exists(FLAGS.file):
      raise ValueError("File does not exist: %s" % input_file)

    output_file = None
    if FLAGS.file_out is not None and (not params.enable_horovod or hvd.rank() == 0):
      output_file = os.path.abspath(FLAGS.file_out)
      tf.logging.info("File output specified: %s" % output_file)

    translate_file(params, estimator, subtokenizer, input_file, output_file, False, FLAGS.report_throughput)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Model arguments
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/tmp/data/translate_ende",
      help="[default: %(default)s] Directory where vocab file is stored.",
      metavar="<DD>")
  parser.add_argument(
      "--vocab_file", "-vf", type=str, default=_VOCAB_FILE,
      help="[default: %(default)s] Name of vocabulary file.",
      metavar="<vf>")
  parser.add_argument(
      "--model_dir", "-md", type=str, default="/tmp/transformer_model",
      help="[default: %(default)s] Directory containing Transformer model "
           "checkpoints.",
      metavar="<MD>")
  parser.add_argument(
      "--params", "-p", type=str, default="big", choices=["base", "big"],
      help="[default: %(default)s] Parameter used for trained model.",
      metavar="<P>")

  # Flags for specifying text/file to be translated.
  parser.add_argument(
      "--text", "-t", type=str, default=None,
      help="[default: %(default)s] Text to translate. Output will be printed "
           "to console.",
      metavar="<T>")
  parser.add_argument(
      "--file", "-f", type=str, default=None,
      help="[default: %(default)s] File containing text to translate. "
           "Translation will be printed to console and, if --file_out is "
           "provided, saved to an output file.",
      metavar="<F>")
  parser.add_argument(
      "--file_out", "-fo", type=str, default=None,
      help="[default: %(default)s] If --file flag is specified, save "
           "translation to this file.",
      metavar="<FO>")
  parser.add_argument(
      "--enable_horovod", "-enable_hvd", action="store_true",
      help="Enable multi-gpu training with horovod")
  parser.add_argument(
      "--enable_fp16", "-enable_fp16", action="store_true",
      help="Enable mixed-precision, fp16 where possible.")
  parser.add_argument(
      "--report_throughput", "-rt", action="store_true",
      help="Report throughput in alternative format")
  parser.add_argument(
    "--sentencepiece", "-sp", action='store_true',
    help="Use SentencePiece tokenizer. Warning: In order to use SP "
          "you have to preprocess dataset with SP as well")
  parser.add_argument(
      "--batch_size", "-b", type=int, default=0,
      help="Override default batch size parameter in prams",
      metavar="<B>")

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
