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

import argparse

DEFAULT_DIR = '/outbrain'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        help='Path with the data required for NVTabular preprocessing. '
             'If stats already exists under metadata_path preprocessing phase will be skipped.',
        type=str,
        default=f'{DEFAULT_DIR}/orig',
        nargs='+'
    )
    parser.add_argument(
        '--metadata_path',
        help='Path with preprocessed NVTabular stats',
        type=str,
        default=f'{DEFAULT_DIR}/data',
        nargs='+'
    )
    parser.add_argument(
        '--tfrecords_path',
        help='Path where converted tfrecords will be stored',
        type=str,
        default=f'{DEFAULT_DIR}/tfrecords',
        nargs='+'
    )
    parser.add_argument(
        '--workers',
        help='Number of TfRecords files to be created',
        type=int,
        default=40
    )

    return parser.parse_args()
