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
import abc
import pathlib
import shutil
from typing import Dict, List

import yaml

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .experiment import ExperimentResult
from .logger import LOGGER
from .stages import ResultsType
from .summary import load_results, save_summary
from .task import Task


class Finalizer(abc.ABC):
    @abc.abstractmethod
    def exec(self, workspace: pathlib.Path, task: Task, results: List[ExperimentResult]):
        pass


class ExperimentFinalizer(Finalizer):
    """
    Public runner finalizer object.
    """

    def exec(self, workspace: pathlib.Path, task: Task, results: List[ExperimentResult]):
        results_path = workspace / task.results_dir

        self._generate_summary(results_path, results)
        self._finalize_task(results_path, task)

    def _finalize_task(self, results_path: pathlib.Path, task: Task) -> None:
        """
        Finalize task information

        Args:
            task: Task object

        Returns:
            None
        """
        task.end()

        file_path = results_path / task.filename

        LOGGER.debug(f"Saving task details to file {file_path}")
        task.to_file(file_path)
        LOGGER.debug("Done")

        LOGGER.info(f"Task details and results stored in {results_path}")

    def _generate_summary(self, results_path: pathlib.Path, experiment_results: List[ExperimentResult]):
        """
        Generate summary for results collected in all experiments

        Args:
            results_path: Path where results should be stored
            experiment_results: Results collected from experiments

        Returns:

        """
        performance_offline_results = list()
        performance_online_results = list()
        results_mapping = {
            ResultsType.TRITON_PERFORMANCE_OFFLINE: performance_offline_results,
            ResultsType.TRITON_PERFORMANCE_ONLINE: performance_online_results,
        }

        self._collect_summary_results(experiment_results, results_mapping)
        self._prepare_final_results(results_path, results_mapping)

    def _collect_summary_results(self, experiment_results: List[ExperimentResult], results_mapping: Dict):
        for experiment_result in experiment_results:
            experiment = experiment_result.experiment
            for result_type, result_path in experiment_result.results.items():

                if not result_path.is_file() and not result_path.is_dir():
                    raise FileNotFoundError(f"Expected file {result_path} not found")

                LOGGER.debug(f"Found {result_type} in {result_path} file.")

                if result_type not in results_mapping:
                    LOGGER.debug(f"Results {result_type} for {experiment.experiment_id} are ignored in final summary.")
                    return

                LOGGER.debug(f"Collecting {result_type} results from {result_path} for summary")
                result = load_results(
                    results_path=result_path,
                    parameters=experiment.parameters,
                    result_type=result_type,
                )

                results_mapping[result_type].extend(result)
                LOGGER.debug(f"Done.")

    def _prepare_final_results(self, results_path: pathlib.Path, results_mapping: Dict) -> None:
        """
        Prepare summary files for offline and online performance

        Args:
            results_path: Path where results should be stored
            results_mapping: Mapping with results type and collected results for given stage

        Returns:
            None
        """
        for results_type, results in results_mapping.items():
            save_summary(
                result_type=results_type,
                results=results,
                summary_dir=results_path,
            )
