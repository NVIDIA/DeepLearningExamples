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

import os

from data.outbrain.features import HASH_BUCKET_SIZES
from data.outbrain.nvtabular.utils.feature_description import transform_spark_to_nvt


def create_config(args):
    stats_file = os.path.join(args.metadata_path, 'stats_wnd_workflow')
    data_bucket_folder = args.data_path
    output_bucket_folder = args.metadata_path
    output_train_folder = os.path.join(output_bucket_folder, 'train/')
    temporary_folder = os.path.join('/tmp', 'preprocessed')
    train_path = os.path.join(temporary_folder, 'train_gdf.parquet')
    valid_path = os.path.join(temporary_folder, 'valid_gdf.parquet')
    output_valid_folder = os.path.join(output_bucket_folder, 'valid/')
    tfrecords_path = args.tfrecords_path
    workers = args.workers
    hash_spec = {transform_spark_to_nvt(column): hash for column, hash in HASH_BUCKET_SIZES.items()}

    config = {
        'stats_file': stats_file,
        'data_bucket_folder': data_bucket_folder,
        'output_bucket_folder': output_bucket_folder,
        'output_train_folder': output_train_folder,
        'temporary_folder': temporary_folder,
        'train_path': train_path,
        'valid_path': valid_path,
        'output_valid_folder': output_valid_folder,
        'tfrecords_path': tfrecords_path,
        'workers': workers,
        'hash_spec': hash_spec
    }

    return config
