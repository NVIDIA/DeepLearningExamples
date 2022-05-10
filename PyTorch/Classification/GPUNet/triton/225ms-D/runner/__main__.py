# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import pathlib
from typing import List

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ...runner.config import Config
from ...runner.executor import Executor
from ...runner.finalizer import ExperimentFinalizer
from ...runner.maintainer import DockerMaintainer
from ...runner.preparer import ExperimentPreparer
from ...runner.runner_proxy import RunnerProxy
from .pipeline_impl import pipeline


class ExperimentRunner(RunnerProxy):
    """
    Experiment Runner proxy for runner wrapper
    """

    maintainer_cls = DockerMaintainer
    executor_cls = Executor
    preparer_cls = ExperimentPreparer
    finalizer_cls = ExperimentFinalizer


def execute(config_path: str, devices: List[str]):
    if len(devices) == 0:
        devices = ["0"]

    config = Config.from_file(config_path)
    runner = ExperimentRunner(config=config, pipeline=pipeline, devices=devices)
    runner.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to configuration file with details.")
    parser.add_argument(
        "--devices", type=str, nargs="*", required=False, help="Path to configuration file with details."
    )

    args = parser.parse_args()

    config_path = args.config_path
    devices = args.devices

    execute(config_path, devices)