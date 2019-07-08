from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import sentencepiece as spm
import process_data as pd

def main(unused_argv):
  print("Preprocessing and saving data")
  sp_bpe = spm.SentencePieceProcessor()
  sp_bpe.load('{}.model'.format(FLAGS.vocab_file))
  compile_data = FLAGS.input_data, FLAGS.target_data
  train_tfrecord_files = pd.encode_and_save_files(
      sp_bpe, FLAGS.data_dir, compile_data, FLAGS.data_tag,
      10)
  for fname in train_tfrecord_files:
    pd.shuffle_records(fname)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/tmp/translate_ende",
      help="[default: %(default)s] Directory for where the "
           "dataset is saved.",
      metavar="<DD>")
  parser.add_argument(
      "--vocab_file", "-vf", type=str,
      help="Name of vocabulary file.",
      metavar="<vf>")
  parser.add_argument(
      "--input_data", "-id", type=str,
      help="Path of the raw data of input language.",
      metavar="<ID>")
  parser.add_argument(
    "--target_data", "-td", type=str,
    help=" Path of the raw data of target language.",
    metavar="<TD>")
  parser.add_argument(
    "--data_tag", "-dt", type=str, default="train",
    help="[default: %(default)s] name tag of TFRecord data.",
    metavar="<DT>")

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)

