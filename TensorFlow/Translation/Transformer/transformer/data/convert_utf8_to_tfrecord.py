from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys

import six
import tensorflow as tf



def convert_utf8_tfrecord(data_dir):
  """Save data from files as encoded Examples in TFrecord format.

  Args:
    data_dir: The directory in which to write the examples

  Returns:
    List of all files produced.
  """
  print('Converting utf8 files to tfrecord')

  src_files = glob.glob(data_dir + '/*.src', recursive=True)
  dst_files = glob.glob(data_dir + '/*.dst', recursive=True)

  src_files.sort()
  dst_files.sort()

  file_pairs = dict(zip(src_files, dst_files))

  counter = 0
  for file_pair in file_pairs:
    # Write examples in tfrecords
    output_path = '/'.join(file_pair.split('/')[:-2])
    output_file = file_pair[:-4].split('/')[-1]
    writer = tf.python_io.TFRecordWriter(output_path + '/' + output_file)

    with open(file_pair, mode='r', newline='\n') as src_seqs:
      with open(file_pairs[file_pair], mode='r', newline='\n') as dst_seqs:
        for src_seq, dst_seq in zip(src_seqs, dst_seqs):
          if src_seq != '\n' and dst_seq != '\n':
            example = dict_to_example({"inputs": [int(x) for x in src_seq.strip().split(' ')],
                                     "targets": [int(x) for x in dst_seq.strip().split(' ')]})

            writer.write(example.SerializeToString())

            if counter > 0 and counter % 100000 == 0:
              tf.logging.info("\tSaving case %d." % counter)

            counter += 1

    writer.close()

  tf.logging.info("Saved %d Examples", counter)


def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  convert_utf8_tfrecord(FLAGS.data_dir)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/research/transformer/processed_data",
      help="[default: %(default)s] Directory for where the "
           "translate_ende_wmt32k dataset is saved.",
      metavar="<DD>")

  FLAGS, unparsed = parser.parse_known_args()
  main(sys.argv)
