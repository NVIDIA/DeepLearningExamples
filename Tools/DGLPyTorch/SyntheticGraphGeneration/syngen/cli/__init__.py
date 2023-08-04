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

from syngen.cli.commands.synthesize import SynthesizeCommand
from syngen.cli.commands.preprocess import PreprocessingCommand
from syngen.cli.commands.mimic_dataset import MimicDatasetCommand
from syngen.cli.commands.pretrain import PretrainCommand


def get_parser():
    parser = argparse.ArgumentParser(
        description="Synthetic Graph Generation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    command = parser.add_subparsers(title="command")
    command.required = True

    SynthesizeCommand().init_parser(command)
    PreprocessingCommand().init_parser(command)
    MimicDatasetCommand().init_parser(command)
    PretrainCommand().init_parser(command)

    return parser
