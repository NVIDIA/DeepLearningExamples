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
import tensorflow as tf

from sim.data.feature_spec import FeatureSpec

# Docker image sets it to "python" for NVTabular purposes (bugfix), which slows down the script 20x
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


def _int64_feature(value, islist=False):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not islist:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def process_chunk(df, sequential_data_start):
    feature_values_lists = [df.iloc[:, i].values for i in range(sequential_data_start)]

    for i in range(sequential_data_start, df.shape[1]):
        values = df.iloc[:, i].values.tolist()
        feature_values_lists.append(values)

    return zip(*feature_values_lists)


def prepare_record(sample, all_feature_names, sequential_data_start):

    feature = {}
    for idx, (f_name, data) in enumerate(zip(all_feature_names, sample)):
        islist = idx >= sequential_data_start
        feature[f_name] = _int64_feature(data, islist)

    record_bytes = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    return record_bytes


def create_default_feature_spec(user_features_cardinalities, item_features_cardinalities,
                                max_seq_len, tfrecord_output_dir, train_output_file, test_output_file):

    train_output = tfrecord_output_dir / train_output_file
    test_output = tfrecord_output_dir / test_output_file

    f_spec = FeatureSpec.get_default_feature_spec(user_features_cardinalities, item_features_cardinalities,
                                                  max_seq_len, train_output, test_output)

    save_path = tfrecord_output_dir / 'feature_spec.yaml'
    f_spec.to_yaml(save_path)


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
    required=True,
    help="number of user specific features.",
    type=int
)
@click.option(
    "--max_seq_len",
    default=100,
    help="maximum possible length of history. (Entries will be padded to that length later)."
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
    help="name of directory within amazon dataset directory containing train data."
)
@click.option(
    "--test_split_dir",
    default='test',
    help="name of directory within amazon dataset directory containing test data."
)
@click.option(
    "--metadata_file",
    default='metadata.json',
    help="name of metadata file within amazon dataset directory (containing feature cardinalities)."
)
@click.option(
    "--train_output_file",
    default='train.tfrecord',
    help='name of train file within output directory.',
    type=str
)
@click.option(
    "--test_output_file",
    default='test.tfrecord',
    help='name of test file within output directory.',
    type=str
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
        train_output_file: str,
        test_output_file: str
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

    os.makedirs(tfrecord_output_dir, exist_ok=True)
    output_splits = [
        tfrecord_output_dir / train_output_file,
        tfrecord_output_dir / test_output_file
    ]

    with open(amazon_dataset_path / metadata_file, 'r') as file:
        metadata = json.load(file)

    feature_cardinalities = []
    for cardinality in metadata['cardinalities']:
        feature_cardinalities.append(cardinality['value'])

    user_features_cardinalities = feature_cardinalities[:number_of_user_features]
    item_features_cardinalities = feature_cardinalities[number_of_user_features:]

    create_default_feature_spec(user_features_cardinalities, item_features_cardinalities, max_seq_len,
                                tfrecord_output_dir, train_output_file, test_output_file)

    number_of_item_features = len(item_features_cardinalities)
    sequential_data_start = 1 + number_of_user_features + number_of_item_features
    all_feature_names = FeatureSpec.get_default_features_names(number_of_user_features, number_of_item_features)
    prepare_record_function = partial(prepare_record, all_feature_names=all_feature_names,
                                      sequential_data_start=sequential_data_start)

    for input_dir, output_file in zip(input_splits, output_splits):

        files = input_dir.glob("part.*.parquet")
        def num_order(p): return int(p.name.split(".")[1])
        paths = sorted(files, key=num_order)

        logging.info(f"Started conversion, will output to {output_file}")

        with tf.io.TFRecordWriter(str(output_file)) as file_writer:
            with multiprocessing.Pool(n_proc) as pool:
                for path in paths:
                    df = pd.read_parquet(path)

                    zipped_data = process_chunk(df, sequential_data_start)

                    records = pool.map(prepare_record_function, zipped_data)
                    for record_bytes in records:
                        file_writer.write(record_bytes)

                    logging.info(f"Processed {path}")


if __name__ == "__main__":
    main()
