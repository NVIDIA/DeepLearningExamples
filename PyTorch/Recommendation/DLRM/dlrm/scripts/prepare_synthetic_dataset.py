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
import torch

from dlrm.data.datasets import SyntheticDataset
from dlrm.data.factories import create_synthetic_datasets
from dlrm.data.utils import write_dataset_to_disk, get_categorical_feature_sizes
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_numerical_features", 13,
                     "Number of numerical features in the dataset. Defaults to 13 for the Criteo Terabyte Dataset")
flags.DEFINE_integer("synthetic_dataset_num_entries",
                     default=int(32768 * 1024),  # 1024 batches for single-GPU training by default
                     help="Number of samples per epoch for the synthetic dataset")
flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10 ** 5)]),
                  help="Embedding table sizes to use with the synthetic dataset")
flags.DEFINE_string("synthetic_dataset_dir", default="/tmp/dlrm_synthetic_data",
                    help="Destination of the saved synthetic dataset")
flags.DEFINE_integer("seed", default=12345, help="Set a seed for generating synthetic data")

def main(argv):
    torch.manual_seed(FLAGS.seed)

    table_sizes = [int(s) for s in FLAGS.synthetic_dataset_table_sizes]
    train_dataset = SyntheticDataset(
        num_entries=FLAGS.synthetic_dataset_num_entries,
        numerical_features=FLAGS.num_numerical_features,
        categorical_feature_sizes=table_sizes
    )
    test_dataset = SyntheticDataset(
        num_entries=FLAGS.synthetic_dataset_num_entries,
        numerical_features=FLAGS.num_numerical_features,
        categorical_feature_sizes=table_sizes
    )

    write_dataset_to_disk(
        FLAGS.synthetic_dataset_dir,
        train_dataset,
        test_dataset,
        FLAGS.synthetic_dataset_table_sizes
    )


if __name__ == '__main__':
    app.run(main)
