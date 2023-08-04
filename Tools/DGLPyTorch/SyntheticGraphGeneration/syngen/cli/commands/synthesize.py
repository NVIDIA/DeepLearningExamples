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
import json

from syngen.cli.commands.base_command import BaseCommand

from syngen.configuration.configuration import SynGenConfiguration
from syngen.synthesizer.configuration_graph_synthesizer import ConfigurationGraphSynthesizer


class SynthesizeCommand(BaseCommand):

    def init_parser(self, base_parser):
        synthesizer = base_parser.add_parser(
            "synthesize",
            help="Run Graph Synthesizer",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        synthesizer.set_defaults(action=self.run)

        synthesizer.add_argument(
            "-cp", "--config-path", type=str, default=None, help="Path to SynGen Configuration file"
        )
        synthesizer.add_argument(
            "--timer-path", type=str, default=None,
            help="Saves generation process timings to the specified file"
        )
        synthesizer.add_argument(
            "-sp", "--save-path", type=str, default="./generated", required=False,
            help="Save path to dump generated files",
        )
        synthesizer.add_argument(
            "--cpu", action='store_true',
            help="Runs all operations on CPU. [Attention] Alignment is not available on CPU"
        )
        synthesizer.add_argument(
            "-v", "--verbose", action='store_true',
            help="Displays generation process progress"
        )

    def run(self, args):
        dict_args = vars(args)

        config_path = dict_args.pop('config_path')
        gpu = not dict_args.pop('cpu')

        with open(config_path, 'r') as f:
            configuration = json.load(f)
        configuration = SynGenConfiguration(configuration)

        synthesizer = ConfigurationGraphSynthesizer(
            configuration,
            gpu=gpu,
            **dict_args,
        )
        synthesizer.fit()
        synthesizer.generate(return_data=False)
