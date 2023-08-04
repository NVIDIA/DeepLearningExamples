# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import logging

from syngen.cli.commands.base_command import BaseCommand
from syngen.preprocessing.datasets import DATASETS

logger = logging.getLogger(__name__)
log = logger


class PreprocessingCommand(BaseCommand):

    def init_parser(self, base_parser):
        preprocessing_parser = base_parser.add_parser(
            "preprocess",
            help="Run Dataset Preprocessing",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        preprocessing_parser.set_defaults(action=self.run)

        preprocessing_parser.add_argument(
            "--dataset", type=str, default=None, required=True, choices=list(DATASETS.keys()),
            help="Dataset to preprocess",
        )
        preprocessing_parser.add_argument(
            "-sp", "--source-path", type=str, default=None, required=True,
            help="Path to raw data",
        )
        preprocessing_parser.add_argument(
            "-dp", "--destination-path", type=str, default=None, required=False,
            help="Path to store the preprocessed data. Default is $source_path/syngen_preprocessed",
        )
        preprocessing_parser.add_argument(
            "--download",
            action='store_true',
            help="Downloads the dataset if specified",
        )
        preprocessing_parser.add_argument(
            "--cpu",
            action='store_true',
            help='Performs the preprocessing_parser without leveraging GPU'
        )
        preprocessing_parser.add_argument(
            "--use-cache",
            action='store_true',
            help='Does nothing if the target preprocessed dataset exists'
        )

        for preprocessing_class in DATASETS.values():
            preprocessing_class.add_cli_args(preprocessing_parser)

    def run(self, args):
        dict_args = vars(args)

        dataset_name = dict_args.pop('dataset')
        source_path = dict_args.pop('source_path')
        destination_path = dict_args.pop('destination_path')
        download = dict_args.pop('download')

        gpu = not dict_args.pop('cpu')
        use_cache = dict_args.pop('use_cache')

        preprocessing_class = DATASETS[dataset_name]

        if download:
            try:
                preprocessing_class(source_path=source_path,
                                    destination_path=destination_path,
                                    download=download,
                                    **dict_args)
                log.info(f"{dataset_name} successfully downloaded into {source_path}")
            except NotImplementedError:
                log.info(f"{dataset_name} does not support automatic downloading, please download the dataset manually")
        else:
            preprocessing = preprocessing_class(source_path=source_path,
                                                destination_path=destination_path,
                                                download=download,
                                                **dict_args)
            preprocessing.transform(gpu=gpu, use_cache=use_cache)
            log.info(f"{dataset_name} successfully preprocessed into {preprocessing.destination_path}")
