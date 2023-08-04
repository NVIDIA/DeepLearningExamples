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

import os
import argparse
import json
import logging
from collections import defaultdict

from syngen.cli.commands.base_command import BaseCommand
from syngen.configuration import SynGenDatasetFeatureSpec, SynGenConfiguration
from syngen.generator.tabular import tabular_generators_classes

from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


class MimicDatasetCommand(BaseCommand):

    def init_parser(self, base_parser):
        mimic_parser = base_parser.add_parser(
            "mimic-dataset",
            help="Quickly creates a SynGen Configuration for the given dataset",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        mimic_parser.set_defaults(action=self.run)

        mimic_parser.add_argument(
            "-dp", "--dataset-path", type=str, required=True,
            help="Path to the dataset in SynGen format"
        )
        mimic_parser.add_argument(
            "-of", "--output-file", type=str, required=True,
            help="Path to the generated SynGen Configuration"
        )
        mimic_parser.add_argument(
            "-tg", "--tab-gen", type=str, choices=list(tabular_generators_classes.keys()), default='kde',
            help="Tabular Generator to mimic all tabular features"
        )
        mimic_parser.add_argument(
            "-rsg", "--random-struct-gen",  action='store_true',
            help="Generates random structure based on Erdos-Renyi model"
        )
        mimic_parser.add_argument(
            "-es", "--edge-scale", type=float, default=None,
            help="Multiples the number of edges to generate by the provided number"
        )
        mimic_parser.add_argument(
            "-en", "--node-scale", type=float, default=None,
            help="Multiples the number of nodes to generate by the provided number"
        )
        mimic_parser.add_argument(
            "-gdp", "--gen-dump-path", type=str, default=None,
            help="Path to store the fitted generators"
        )

    def run(self, args):

        dict_args = vars(args)
        feature_spec = SynGenDatasetFeatureSpec.instantiate_from_preprocessed(dict_args['dataset_path'])

        scales = {
            MetaData.EDGES: dict_args['edge_scale'],
            MetaData.NODES: dict_args['node_scale'],
        }

        for part in [MetaData.NODES, MetaData.EDGES]:
            for part_info in feature_spec[part]:

                if scales[part]:
                    part_info[MetaData.COUNT] = int(part_info[MetaData.COUNT] * scales[part])

                if MetaData.FEATURES in part_info and len(part_info[MetaData.FEATURES]) > 0:

                    feature_files_content = defaultdict(list)

                    for feature in part_info[MetaData.FEATURES]:
                        if MetaData.FEATURE_FILE in feature:
                            feature_files_content[feature[MetaData.FEATURE_FILE]].append(feature[MetaData.NAME])

                    if feature_files_content:
                        part_info[MetaData.TABULAR_GENERATORS] = [
                            {
                                MetaData.TYPE: dict_args['tab_gen'],
                                MetaData.FEATURES_LIST: feats_list,
                                MetaData.FEATURE_FILE: ff,
                                MetaData.DATA_SOURCE: {
                                    MetaData.TYPE: 'rnd',
                                } if dict_args['tab_gen'] == 'random'
                                else
                                {
                                    MetaData.TYPE: 'cfg',
                                    MetaData.PATH: dict_args['dataset_path'],
                                    MetaData.NAME: part_info[MetaData.NAME],
                                },
                                MetaData.PARAMS: {},
                                MetaData.DUMP_PATH: os.path.join(dict_args['gen_dump_path'],
                                                                 f"{part}_{part_info[MetaData.NAME]}_tab_gen_{idx}.pkl")
                                if dict_args['gen_dump_path'] else None
                            }
                            for idx, (ff, feats_list) in enumerate(feature_files_content.items())
                        ]
                    else:
                        part_info[MetaData.TABULAR_GENERATORS] = [
                            {
                                MetaData.TYPE: dict_args['tab_gen'],
                                MetaData.FEATURES_LIST: -1,
                                MetaData.DATA_SOURCE: {
                                    MetaData.TYPE: 'rnd',
                                } if dict_args['tab_gen'] == 'random'
                                else
                                {
                                    MetaData.TYPE: 'cfg',
                                    MetaData.PATH: dict_args['dataset_path'],
                                    MetaData.NAME: part_info[MetaData.NAME],
                                },
                                MetaData.PARAMS: {},
                                MetaData.DUMP_PATH: os.path.join(dict_args['gen_dump_path'],
                                                                 f"{part}_{part_info[MetaData.NAME]}_tab_gen_{0}.pkl")
                                if dict_args['gen_dump_path'] else None
                            }
                        ]
                if part == MetaData.EDGES:
                    part_info[MetaData.STRUCTURE_GENERATOR] = {
                        MetaData.TYPE: 'RMAT',
                        MetaData.DATA_SOURCE: {
                            MetaData.TYPE: 'rnd',
                        } if dict_args['random_struct_gen']
                        else
                        {
                            MetaData.TYPE: 'cfg',
                            MetaData.PATH: dict_args['dataset_path'],
                            MetaData.NAME: part_info[MetaData.NAME],
                        },
                        MetaData.PARAMS: {},
                        MetaData.DUMP_PATH: os.path.join(dict_args['gen_dump_path'],
                                                         f"{part_info[MetaData.NAME]}_struct_gen.pkl")
                        if dict_args['gen_dump_path'] else None
                    }

        config = SynGenConfiguration(feature_spec)

        with open(dict_args['output_file'], 'w') as f:
            json.dump(config, f, indent=4)

        log.info(f"SynGen Configuration saved into {dict_args['output_file']}")
