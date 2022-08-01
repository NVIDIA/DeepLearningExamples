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
import pathlib
from typing import Dict, Tuple

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .stages import (
    ConversionStage,
    DeployStage,
    ExportStage,
    ResultsType,
    TritonPerformanceOfflineStage,
    TritonPerformanceOnlineStage,
    TritonPreparePerformanceProfilingDataStage,
)


class Pipeline:
    """
    Definition of stages that has to be executed before and during experiments
    """

    # Stages to execute as part of single experiment
    _experiment_stages = [
        ExportStage.label,
        ConversionStage.label,
        DeployStage.label,
        TritonPreparePerformanceProfilingDataStage.label,
        TritonPerformanceOfflineStage.label,
        TritonPerformanceOnlineStage.label,
    ]

    def __init__(self):
        """
        Initialize pipeline
        """
        self._stages: Dict = dict()

    def model_export(self, commands: Tuple[str, ...]) -> None:
        """
        Model export stage

        Args:
            commands: Commands to be executed as part of stage

        Returns:
            None
        """
        stage = ExportStage(commands=commands)
        self._stages[stage.label] = stage

    def model_conversion(self, commands: Tuple[str, ...]) -> None:
        """
        Model conversion stage

        Args:
            commands: Commands to be executed as part of stage

        Returns:
            None
        """
        stage = ConversionStage(commands=commands)
        self._stages[stage.label] = stage

    def model_deploy(self, commands: Tuple[str, ...]) -> None:
        """
        Model deployment stage

        Args:
            commands: Commands to be executed as part of stage

        Returns:
            None
        """
        stage = DeployStage(commands=commands)
        self._stages[stage.label] = stage

    def triton_prepare_performance_profiling_data(self, commands: Tuple[str, ...]) -> None:
        """
        Model profiling data creation stage

        Args:
            commands: Commands to be executed as part of stage

        Returns:
            None
        """
        stage = TritonPreparePerformanceProfilingDataStage(commands=commands)
        self._stages[stage.label] = stage

    def triton_performance_offline_tests(self, commands: Tuple[str, ...], result_path: str) -> None:
        """
        Model performance offline test stage

        Args:
            commands: Commands to be executed as part of stage
            result_path: Path where results file is stored

        Returns:
            None
        """
        stage = TritonPerformanceOfflineStage(
            commands=commands,
            result_path=result_path,
            result_type=ResultsType.TRITON_PERFORMANCE_OFFLINE,
        )
        self._stages[stage.label] = stage

    def triton_performance_online_tests(self, commands: Tuple[str, ...], result_path: str) -> None:
        """
        Model performance online test stage

        Args:
            commands: Commands to be executed as part of stage
            result_path: Path where results file is stored

        Returns:
            None
        """
        stage = TritonPerformanceOnlineStage(
            commands=commands,
            result_path=result_path,
            result_type=ResultsType.TRITON_PERFORMANCE_ONLINE,
        )
        self._stages[stage.label] = stage

    def stages(self):
        """
        Generate stages which should be run per experiment

        Returns:
            Generator with stages object
        """
        for stage_name in self._experiment_stages:
            stage = self._stages.get(stage_name)
            if not stage:
                continue

            yield stage
