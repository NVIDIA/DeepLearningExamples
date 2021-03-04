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

os.environ['TF_MEMORY_ALLOCATION'] = "0.0"
from data.outbrain.nvtabular.utils.converter import nvt_to_tfrecords
from data.outbrain.nvtabular.utils.workflow import execute_pipeline
from data.outbrain.nvtabular.utils.arguments import parse_args
from data.outbrain.nvtabular.utils.setup import create_config


def is_empty(path):
    return not os.path.exists(path) or (not os.path.isfile(path) and not os.listdir(path))


def main():
    args = parse_args()
    config = create_config(args)
    if is_empty(args.metadata_path):
        logging.warning('Creating new stats data into {}'.format(config['stats_file']))
        execute_pipeline(config)
    else:
        logging.warning('Directory is not empty {args.metadata_path}')
        logging.warning('Skipping NVTabular preprocessing')

    if os.path.exists(config['output_train_folder']) and os.path.exists(config['output_valid_folder']):
        if is_empty(config['tfrecords_path']):
            logging.warning('Executing NVTabular parquets to TFRecords conversion')
            nvt_to_tfrecords(config)
        else:
            logging.warning(f"Directory is not empty {config['tfrecords_path']}")
            logging.warning('Skipping TFrecords conversion')
    else:
        logging.warning(f'Train and validation dataset not found in {args.metadata_path}')


if __name__ == '__main__':
    main()
