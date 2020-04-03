# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import sys

import outbrain_transform

import tensorflow as tf
import glob

import pandas as pd

import trainer.features


def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments including program name.
  Returns:
    The parsed arguments as returned by argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser(
      description='Runs Transformation on the Outbrain Click Prediction model data.')

  parser.add_argument(
      '--training_data',
      default='',
      help='Data to analyze and encode as training features.')
  parser.add_argument(
      '--eval_data',
      default='',
      help='Data to encode as evaluation features.')
  parser.add_argument(
      '--output_dir',
      default=None,
      required=True,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))
  parser.add_argument('--batch_size', default=None, type=int, help='Size of batches to create.')
  parser.add_argument('--submission', default=False, action='store_true', help='Use real test set for submission')

  args, _ = parser.parse_known_args(args=argv[1:])

  return args

# a version of this method that prefers pandas methods
def local_transform_chunk(nr, csv, output_prefix, min_logs, max_logs, batch_size=None, remainder=None):
  # put any remainder at the front of the line, with the new rows after
  if remainder is not None:
    csv = remainder.append(csv)
  
  # for each batch, slice into the datafrom to retrieve the corresponding data
  print(str(datetime.datetime.now()) + '\tWriting rows...')
  num_rows = len(csv.index)
  with tf.python_io.TFRecordWriter(output_prefix + str(nr).zfill(3) + '.tfrecord') as writer:
    for start_ind in range(0,num_rows,batch_size if batch_size is not None else 1): # for each batch
      if start_ind + batch_size - 1 > num_rows: # if we'd run out of rows
        return csv.iloc[start_ind:] # return remainder for use with the next file
      # otherwise write this batch to TFRecord
      csv_slice = csv.iloc[start_ind:start_ind+(batch_size if batch_size is not None else 1)]
      example = outbrain_transform.create_tf_example(csv_slice, min_logs, max_logs)
      writer.write(example.SerializeToString())

# calculate min and max stats for the given dataframes all in one go
def compute_min_max_logs(dataframes):
  print(str(datetime.datetime.now()) + '\tComputing min and max')
  min_logs = {}
  max_logs = {}
  df = pd.concat(dataframes) # concatenate all dataframes, to process at once
  for name in trainer.features.FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
    feature_series = df[name]
    min_logs[name + '_log_01scaled'] = outbrain_transform.log2_1p(feature_series.min(axis=0)*1000)
    max_logs[name + '_log_01scaled'] = outbrain_transform.log2_1p(feature_series.max(axis=0)*1000)
  for name in trainer.features.INT_COLUMNS:
    feature_series = df[name]
    min_logs[name + '_log_01scaled'] = outbrain_transform.log2_1p(feature_series.min(axis=0))
    max_logs[name + '_log_01scaled'] = outbrain_transform.log2_1p(feature_series.max(axis=0))
  return min_logs, max_logs


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)
  
  # Retrieve and sort training and eval data (to ensure consistent order)
  # Order is important so that the right data will get sorted together for MAP
  training_data = sorted(glob.glob(args.training_data))
  eval_data = sorted(glob.glob(args.eval_data))
  print('Training data:\n{}\nFound:\n{}'.format(args.training_data,training_data))
  print('Evaluation data:\n{}\nFound:\n{}'.format(args.eval_data,eval_data))
  
  outbrain_transform.make_spec(args.output_dir + '/transformed_metadata', batch_size=args.batch_size)
  
  # read all dataframes
  print('\n' + str(datetime.datetime.now()) + '\tReading input files')
  eval_dataframes = [pd.read_csv(filename, header=None, names=outbrain_transform.CSV_ORDERED_COLUMNS) 
                for filename in eval_data]
  train_dataframes = [pd.read_csv(filename, header=None, names=outbrain_transform.CSV_ORDERED_COLUMNS) 
                for filename in training_data]
  
  # calculate stats once over all records given
  min_logs, max_logs = compute_min_max_logs(eval_dataframes + train_dataframes)
 
  if args.submission:
    train_output_string = '/sub_train_'
    eval_output_string = '/test_'
  else:
    train_output_string = '/train_'
    eval_output_string = '/eval_'
 
  # process eval files
  print('\n' + str(datetime.datetime.now()) + '\tWorking on evaluation data')
  eval_remainder = None # remainder when a file's records don't divide evenly into batches
  for i, df in enumerate(eval_dataframes):
    print(eval_data[i])
    eval_remainder = local_transform_chunk(i, df, args.output_dir + eval_output_string, min_logs, max_logs,
                                           batch_size=args.batch_size, remainder=eval_remainder)
  if eval_remainder is not None:
    print('Dropping {} records (eval) on the floor'.format(len(eval_remainder)))
  
  # process train files
  print('\n' + str(datetime.datetime.now()) + '\tWorking on training data')
  train_remainder = None
  for i, df in enumerate(train_dataframes):
    print(training_data[i])
    train_remainder = local_transform_chunk(i, df, args.output_dir + train_output_string, min_logs, max_logs,
                                           batch_size=args.batch_size, remainder=train_remainder)
  if train_remainder is not None:
    print('Dropping {} records (train) on the floor'.format(len(train_remainder)))

if __name__ == '__main__':
  main()
