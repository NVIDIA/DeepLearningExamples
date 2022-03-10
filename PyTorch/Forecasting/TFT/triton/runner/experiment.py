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


import dataclasses
import pathlib
from datetime import datetime
from typing import Any, Dict, Optional

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .core import DataObject


class ExperimentStatus(object):
    """
    Experiment status flags object
    """

    SUCCEED = "Succeed"
    FAILED = "Failed"


class StageStatus:
    """
    Stages status flags object
    """

    SUCCEED = "Succeed"
    FAILED = "Failed"


class Stage(DataObject):
    """
    Stage data object
    """

    name: str
    status: str
    started_at: Optional[int]
    ended_at: Optional[int]
    result_path: Optional[str]
    result_type: Optional[str]

    def __init__(
        self,
        name: str,
        result_path: Optional[str],
        result_type: Optional[str],
        status: str = StageStatus.FAILED,
        started_at: Optional[int] = None,
        ended_at: Optional[int] = None,
    ):
        """

        Args:
            name: name of stage
            result_path: path where results file is stored
            result_type: type of results
            status: success/fail status
            started_at: time when stage has started
            ended_at: time when stage has ended
        """
        self.name = name
        self.status = status
        self.started_at = started_at
        self.ended_at = ended_at

        self.result_path = result_path
        self.result_type = result_type

    def start(self) -> None:
        """
        Update stage execution info at start

        Returns:
            None
        """
        self.started_at = int(datetime.utcnow().timestamp())

    def end(self) -> None:
        """
        Update stage execution info at end

        Returns:
            None
        """
        self.status = StageStatus.SUCCEED
        self.ended_at = int(datetime.utcnow().timestamp())


class Experiment(DataObject):
    """
    Experiment data object
    """

    experiment_id: int
    parameters: Dict
    stages: Dict[str, Stage]
    results: Dict[str, str]
    status: str
    started_at: Optional[int]
    ended_at: Optional[int]

    def __init__(
        self,
        experiment_id: int,
        parameters: Dict,
        stages: Dict[str, Stage],
        results: Dict[str, str],
        started_at: Optional[int] = None,
        ended_at: Optional[int] = None,
        status: str = ExperimentStatus.FAILED,
    ):
        """
        Args:
            experiment_id: experiment identifier
            parameters: dictionary with experiment configuration
            stages: dictionary with stages run in experiment
            results: mapping between results types and location where are stored
            started_at: time when experiment has started
            ended_at: time when experiment has ended
            status: experiment success/fail information
        """
        self.experiment_id = experiment_id
        self.started_at = started_at
        self.ended_at = ended_at
        self.parameters = parameters
        self.stages = stages
        self.status = status

        self.results = results
        self.results_dir = f"experiment_{experiment_id}"

    def start(self) -> None:
        """
        Update experiment execution info at start

        Returns:
            None
        """
        self.started_at = int(datetime.utcnow().timestamp())

    def end(self) -> None:
        """
        Update experiment execution info at end

        Returns:
            None
        """
        self.status = ExperimentStatus.SUCCEED
        self.ended_at = int(datetime.utcnow().timestamp())


@dataclasses.dataclass
class Status:
    state: ExperimentStatus
    message: str


@dataclasses.dataclass
class ExperimentResult:
    """
    Experiment result object
    """

    status: Status
    experiment: Experiment
    results: Dict[str, pathlib.Path]
    payload: Dict[str, Any] = dataclasses.field(default_factory=dict)
