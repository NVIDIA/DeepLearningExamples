# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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
import tensorflow as tf
import os
import tqdm

from absl import app, flags

from .defaults import DTYPE_SELECTOR, TRAIN_MAPPING, TEST_MAPPING
from .synthetic_dataset import SyntheticDataset
from .feature_spec import FeatureSpec

FLAGS = flags.FLAGS

flags.DEFINE_integer("synthetic_dataset_num_entries",
                     default=int(32768 * 1024),  # 1024 batches for single-GPU training by default
                     help="Number of samples per epoch for the synthetic dataset."
                          "This is rounded down to a multiple of batch size")

flags.DEFINE_integer("synthetic_dataset_batch_size",
                     default=int(32768), help="Batch size - number of unique records")

flags.DEFINE_integer("num_numerical_features", default=13,
                     help="Number of numerical features in the dataset. Defaults to 13 for the Criteo Terabyte Dataset")

flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10 ** 5)]),
                  help="Cardinality of each categorical feature")

flags.DEFINE_string("feature_spec", default=None,
                    help="Feature specification file describing the desired dataset."
                         "Only feature_spec and channel_spec sections are required and used."
                         "Overrides num_numerical_features and synthetic_dataset_table_sizes")

flags.DEFINE_string("synthetic_dataset_dir", default="/tmp/dlrm_synthetic_data",
                    help="Destination of the saved synthetic dataset")

flags.DEFINE_integer("seed", default=12345, help="Set a seed for generating synthetic data")


def write_dataset_to_disk(dataset_train, dataset_test, feature_spec: FeatureSpec) -> None:

    feature_spec.check_feature_spec()  # We rely on the feature spec being properly formatted

    categorical_features_list = feature_spec.get_categorical_feature_names()
    categorical_features_types = [feature_spec.feature_spec[feature_name][DTYPE_SELECTOR]
                                  for feature_name in categorical_features_list]
    number_of_numerical_features = feature_spec.get_number_of_numerical_features()
    number_of_categorical_features = len(categorical_features_list)

    for mapping_name, dataset in zip((TRAIN_MAPPING, TEST_MAPPING),
                                     (dataset_train, dataset_test)):
        file_streams = []
        label_path, numerical_path, categorical_paths = feature_spec.get_mapping_paths(mapping_name)
        try:
            os.makedirs(os.path.dirname(numerical_path), exist_ok=True)
            numerical_f = open(numerical_path, "wb+")
            file_streams.append(numerical_f)

            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            label_f = open(label_path, 'wb+')
            file_streams.append(label_f)

            categorical_fs = []
            for feature_name in categorical_features_list:
                local_path = categorical_paths[feature_name]
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                fs = open(local_path, 'wb+')
                categorical_fs.append(fs)
                file_streams.append(fs)

            pipe = iter(dataset.op())
            for _ in tqdm.tqdm(
                    range(len(dataset)), desc=mapping_name + " dataset saving"):
                (numerical, categorical), label = pipe.get_next()
                categoricals = tf.split(categorical, number_of_categorical_features, axis=1)
                assert (numerical.shape[-1] == number_of_numerical_features)
                assert (len(categoricals) == number_of_categorical_features)

                numerical_f.write(numerical.numpy().astype('float16').tobytes())  # numerical is always float16
                label_f.write(label.numpy().astype('bool').tobytes())  # label is always boolean
                for cat_type, cat_tensor, cat_file in zip(categorical_features_types, categoricals, categorical_fs):
                    cat_file.write(cat_tensor.numpy().astype(cat_type).tobytes())
        finally:
            for stream in file_streams:
                stream.close()
    feature_spec.to_yaml()


def main(argv):
    tf.random.set_seed(FLAGS.seed)

    number_of_entries = FLAGS.synthetic_dataset_num_entries
    batch_size = FLAGS.synthetic_dataset_batch_size
    number_of_batches = number_of_entries // batch_size

    if FLAGS.feature_spec is not None:
        fspec = FeatureSpec.from_yaml(FLAGS.feature_spec)
    else:
        cardinalities = [int(s) for s in FLAGS.synthetic_dataset_table_sizes]
        fspec = FeatureSpec.get_default_feature_spec(number_of_numerical_features=FLAGS.num_numerical_features,
                                                     categorical_feature_cardinalities=cardinalities)

    fspec.base_directory = FLAGS.synthetic_dataset_dir
    fspec.check_feature_spec()

    number_of_numerical_features = fspec.get_number_of_numerical_features()
    categorical_feature_sizes = fspec.get_categorical_sizes()

    train_dataset = SyntheticDataset(batch_size=batch_size, num_numerical_features=number_of_numerical_features,
                                     categorical_feature_cardinalities=categorical_feature_sizes,
                                     num_batches=number_of_batches)

    test_dataset = SyntheticDataset(batch_size=batch_size, num_numerical_features=number_of_numerical_features,
                                    categorical_feature_cardinalities=categorical_feature_sizes,
                                    num_batches=number_of_batches)

    write_dataset_to_disk(
        dataset_train=train_dataset,
        dataset_test=test_dataset,
        feature_spec=fspec
    )


if __name__ == '__main__':
    app.run(main)
