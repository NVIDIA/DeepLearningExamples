# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import logging
import os
from multiprocessing import Process

import pandas as pd
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io

from data.outbrain.features import PREBATCH_SIZE
from data.outbrain.nvtabular.utils.feature_description import transform_nvt_to_spark, CATEGORICAL_COLUMNS, \
    DISPLAY_ID_COLUMN, EXCLUDE_COLUMNS


def create_metadata(df, prebatch_size, output_path):
    fixed_shape = [prebatch_size, 1]
    spec = {}
    for column in df:
        if column in CATEGORICAL_COLUMNS + [DISPLAY_ID_COLUMN]:
            spec[transform_nvt_to_spark(column)] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.int64,
                                                                         default_value=None)
        else:
            spec[transform_nvt_to_spark(column)] = tf.io.FixedLenFeature(shape=fixed_shape, dtype=tf.float32,
                                                                         default_value=None)
    metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(spec))
    metadata_io.write_metadata(metadata, output_path)


def create_tf_example(df, start_index, offset):
    parsed_features = {}
    records = df.loc[start_index:start_index + offset - 1]
    for column in records:
        if column in CATEGORICAL_COLUMNS + [DISPLAY_ID_COLUMN]:
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=records[column].to_numpy()))
        else:
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=records[column].to_numpy()))
        parsed_features[transform_nvt_to_spark(column)] = feature
    features = tf.train.Features(feature=parsed_features)
    return tf.train.Example(features=features)


def create_tf_records(df, prebatch_size, output_path):
    with tf.io.TFRecordWriter(output_path) as out_file:
        start_index = df.index[0]
        for index in range(start_index, df.shape[0] + start_index - prebatch_size + 1, prebatch_size):
            example = create_tf_example(df, index, prebatch_size)
            out_file.write(example.SerializeToString())


def convert(path_to_nvt_dataset, output_path, prebatch_size, exclude_columns, workers=6):
    train_path = os.path.join(path_to_nvt_dataset, 'train')
    valid_path = os.path.join(path_to_nvt_dataset, 'valid')
    output_metadata_path = os.path.join(output_path, 'transformed_metadata')
    output_train_path = os.path.join(output_path, 'train')
    output_valid_path = os.path.join(output_path, 'eval')

    for directory in [output_metadata_path, output_train_path, output_valid_path]:
        os.makedirs(directory, exist_ok=True)

    train_workers, valid_workers = [], []
    output_train_paths, output_valid_paths = [], []

    for worker in range(workers):
        part_number = str(worker).rjust(5, '0')
        record_train_path = os.path.join(output_train_path, f'part-r-{part_number}')
        record_valid_path = os.path.join(output_valid_path, f'part-r-{part_number}')
        output_train_paths.append(record_train_path)
        output_valid_paths.append(record_valid_path)

    logging.warning(f'Prebatch size set to {prebatch_size}')
    logging.warning(f'Number of TFRecords set to {workers}')

    logging.warning(f'Reading training parquets from {train_path}')
    df_train = pd.read_parquet(train_path, engine='pyarrow')
    logging.warning('Done')

    logging.warning(f'Removing training columns {exclude_columns}')
    df_train = df_train.drop(columns=exclude_columns)
    logging.warning('Done')

    logging.warning(f'Creating metadata in {output_metadata_path}')
    metadata_worker = Process(target=create_metadata, args=(df_train, prebatch_size, output_metadata_path))
    metadata_worker.start()

    logging.warning(f'Creating training TFrecords to {output_train_paths}')

    shape = df_train.shape[0] // workers
    shape = shape + (prebatch_size - shape % prebatch_size)

    for worker_index in range(workers):
        df_subset = df_train.loc[worker_index * shape:(worker_index + 1) * shape - 1]
        worker = Process(target=create_tf_records, args=(df_subset, prebatch_size, output_train_paths[worker_index]))
        train_workers.append(worker)

    for worker in train_workers:
        worker.start()

    logging.warning(f'Reading validation parquets from {valid_path}')
    df_valid = pd.read_parquet(valid_path, engine='pyarrow')
    logging.warning('Done')

    logging.warning(f'Removing validation columns {exclude_columns}')
    df_valid = df_valid.drop(columns=exclude_columns)
    logging.warning('Done')

    logging.warning(f'Creating validation TFrecords to {output_valid_paths}')

    shape = df_valid.shape[0] // workers
    shape = shape + (prebatch_size - shape % prebatch_size)

    for worker_index in range(workers):
        df_subset = df_valid.loc[worker_index * shape:(worker_index + 1) * shape - 1]
        worker = Process(target=create_tf_records, args=(df_subset, prebatch_size, output_valid_paths[worker_index]))
        valid_workers.append(worker)

    for worker in valid_workers:
        worker.start()

    for worker_index in range(workers):
        metadata_worker.join()
        train_workers[worker_index].join()
        valid_workers[worker_index].join()

    logging.warning('Done')

    del df_train
    del df_valid

    return output_path


def nvt_to_tfrecords(config):
    path_to_nvt_dataset = config['output_bucket_folder']
    output_path = config['tfrecords_path']
    workers = config['workers']

    convert(
        path_to_nvt_dataset=path_to_nvt_dataset,
        output_path=output_path,
        prebatch_size=PREBATCH_SIZE,
        exclude_columns=EXCLUDE_COLUMNS,
        workers=workers
    )
