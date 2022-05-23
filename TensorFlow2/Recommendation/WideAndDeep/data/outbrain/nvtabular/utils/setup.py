# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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


def create_config(args):
    data_bucket_folder = args.data_path
    output_bucket_folder = args.metadata_path
    temporary_folder = os.path.join("/tmp", "preprocessed")
    train_path = os.path.join(temporary_folder, "train_gdf.parquet")
    valid_path = os.path.join(temporary_folder, "valid_gdf.parquet")
    stats_file = os.path.join(temporary_folder, "stats_wnd_workflow")
    output_train_folder = os.path.join(output_bucket_folder, "train/")
    output_valid_folder = os.path.join(output_bucket_folder, "valid/")
    hash_spec = HASH_BUCKET_SIZES

    config = {
        "stats_file": stats_file,
        "data_bucket_folder": data_bucket_folder,
        "output_bucket_folder": output_bucket_folder,
        "output_train_folder": output_train_folder,
        "temporary_folder": temporary_folder,
        "train_path": train_path,
        "valid_path": valid_path,
        "output_valid_folder": output_valid_folder,
        "hash_spec": hash_spec,
        "dask": args.use_dask,
    }

    return config
