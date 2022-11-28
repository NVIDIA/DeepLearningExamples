# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import multiprocessing
import os
import pathlib
from functools import partial

import click
import pandas as pd
import numpy as np
import tensorflow as tf

from sim.data.feature_spec import FeatureSpec
from sim.data.defaults import TRAIN_MAPPING, TEST_MAPPING, REMAINDER_FILENAME, FILES_SELECTOR

# Docker image sets it to "python" for NVTabular purposes (bugfix), which slows down the script 20x
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

def prepare_record(sample, all_feature_names, sequential_data_start, prebatch):
    feature = {}
    for idx, (f_name, data) in enumerate(zip(all_feature_names, sample.values())):
        if idx >= sequential_data_start:
            if prebatch:
                data = np.array(data).flatten()
        else:
            if not prebatch:
                data = [data]

        feature[f_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=data))

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def save_records(output_path, records, base_output_path, feature_spec, mapping):

    with tf.io.TFRecordWriter(str(output_path)) as file_writer:
        for record_bytes in records:
            file_writer.write(record_bytes)

    feature_spec.source_spec[mapping][0][FILES_SELECTOR].append(
        str(output_path.relative_to(base_output_path))
    )

    logging.info(f'Created: {output_path}')


@click.command()
@click.option(
    "--amazon_dataset_path",
    required=True,
    help="Path to the dataset directory.",
    type=str,
)
@click.option(
    "--tfrecord_output_dir",
    required=True,
    help="Path of directory to output tfrecord files.",
    type=str,
)
@click.option(
    "--number_of_user_features",
    default=1,
    help="number of user specific features. Default is 1 for amazon books dataset (user_id).",
    type=int
)
@click.option(
    "--max_seq_len",
    default=100,
    help="maximum possible length of history. (Entries will be padded to that length later).",
    type=int
)
@click.option(
    "--n_proc",
    default=multiprocessing.cpu_count(),
    help="Number of processes started to speed up conversion to tfrecord.",
    type=int,
)
@click.option(
    "--train_split_dir",
    default='train',
    help="Name of directory within amazon dataset directory containing train data.",
    type=str
)
@click.option(
    "--test_split_dir",
    default='test',
    help="Name of directory within amazon dataset directory containing test data.",
    type=str,
)
@click.option(
    "--metadata_file",
    default='metadata.json',
    help="Name of metadata file within amazon dataset directory (containing feature cardinalities).",
    type=str
)
@click.option(
    "--train_output_dir",
    default='train',
    help="Name of train directory within output directory.",
    type=str
)
@click.option(
    "--test_output_dir",
    default='test',
    help='Name of test directory within output directory.',
    type=str
)
@click.option(
    "--train_parts",
    default=8,
    help="Number of output train files.",
    type=int
)
@click.option(
    "--test_parts",
    default=4,
    help="Number of output test files.",
    type=int
)
@click.option(
    "--prebatch_train_size",
    default=0,
    help='Apply batching to data in preprocessing. If prebatch_size == 0, no prebatching is done.',
    type=int
)
@click.option(
    "--prebatch_test_size",
    default=0,
    help='Apply batching to data in preprocessing. If prebatch_size == 0, no prebatching is done.',
    type=int
)
def main(
        amazon_dataset_path: str,
        tfrecord_output_dir: str,
        number_of_user_features: int,
        max_seq_len: int,
        n_proc: int,
        train_split_dir: str,
        test_split_dir: str,
        metadata_file: str,
        train_output_dir: str,
        test_output_dir: str,
        train_parts: int,
        test_parts: int,
        prebatch_train_size: int,
        prebatch_test_size: int
):
    """
    read_parquet()
    create tf.train.Features
    create default FeatureSpec
    dump to Tfrecords
    """

    amazon_dataset_path = pathlib.Path(amazon_dataset_path)
    tfrecord_output_dir = pathlib.Path(tfrecord_output_dir)

    input_splits = [
        amazon_dataset_path / train_split_dir,
        amazon_dataset_path / test_split_dir
    ]

    output_splits = [
        tfrecord_output_dir / train_output_dir,
        tfrecord_output_dir / test_output_dir
    ]
    for split_dir in output_splits:
        os.makedirs(split_dir, exist_ok=True)

    with open(amazon_dataset_path / metadata_file, 'r') as file:
        metadata = json.load(file)

    feature_cardinalities = []
    for cardinality in metadata['cardinalities']:
        feature_cardinalities.append(cardinality['value'])

    user_features_cardinalities = feature_cardinalities[:number_of_user_features]
    item_features_cardinalities = feature_cardinalities[number_of_user_features:]

    feature_spec = FeatureSpec.get_default_feature_spec(user_features_cardinalities, item_features_cardinalities, max_seq_len)

    number_of_item_features = len(item_features_cardinalities)
    sequential_data_start = 1 + number_of_user_features + number_of_item_features
    all_feature_names = FeatureSpec.get_default_features_names(number_of_user_features, number_of_item_features)
    
    prebatch_per_split = [prebatch_train_size, prebatch_test_size]
    parts_per_split = [train_parts, test_parts]
    mappings = [TRAIN_MAPPING, TEST_MAPPING]

    for mapping, input_dir, output_dir, output_parts, prebatch_size in zip(mappings, input_splits, output_splits, parts_per_split, prebatch_per_split):

        prebatch = prebatch_size > 0
        prepare_record_function = partial(prepare_record, all_feature_names=all_feature_names,
                                        sequential_data_start=sequential_data_start, prebatch=prebatch)
        save_records_function = partial(save_records, base_output_path=tfrecord_output_dir, feature_spec=feature_spec, mapping=mapping)

        logging.info(f"Started conversion, will output to {output_dir}")

        df = pd.read_parquet(input_dir, engine='pyarrow')

        logging.info("Parquet loaded")

        if prebatch:
            df['batch_index'] = df.index // prebatch_size
            df = df.groupby('batch_index').agg(list)
            if len(df.iloc[-1, 0]) < prebatch_size:
                remainder = df[-1:].to_dict('records')[0]
                remainder = prepare_record_function(remainder)

                df = df[:-1]

            logging.info("Prebatching applied")

        df = df.to_dict('records')
        with multiprocessing.Pool(n_proc) as pool:
            records = pool.map(prepare_record_function, df)

        logging.info("Records created")

        records = np.array_split(records, output_parts)
        for i, records_part in enumerate(records):
            if len(records_part) > 0:
                save_records_function(output_dir / f'part_{i}.tfrecord', records_part)

        if prebatch:
            save_records_function(output_dir / REMAINDER_FILENAME, [remainder])

    feature_spec.to_yaml(tfrecord_output_dir / 'feature_spec.yaml')


if __name__ == "__main__":
    main()
