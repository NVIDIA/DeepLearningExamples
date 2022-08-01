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
import pathlib
import signal
import sys
from typing import List, Type

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .config import Config
from .exceptions import RunnerException
from .executor import Executor
from .finalizer import Finalizer
from .logger import LOGGER, log_format
from .maintainer import Maintainer
from .pipeline import Pipeline
from .preparer import Preparer
from .triton import Triton


class Runner:
    """
    Runner class. Main entrypoint to performing task and experiments
    """

    WORKSPACE = pathlib.Path.cwd()
    EXECUTOR_WORKSPACE = WORKSPACE / "runner_workspace"

    def __init__(
        self,
        pipeline: Pipeline,
        config: Config,
        executor_cls: Type[Executor],
        maintainer_cls: Type[Maintainer],
        preparer_cls: Type[Preparer],
        finalizer_cls: Type[Finalizer],
        devices: List[str] = None,
        log_level: int = logging.INFO,
    ):
        self._pipeline = pipeline
        self._config = config

        self._pipeline = pipeline
        self._config = config
        self._preparer = preparer_cls()
        self._finalizer = finalizer_cls()
        self._devices = devices or ["0"]
        self._log_level = log_level
        self._logs_dir = self.EXECUTOR_WORKSPACE / "logs"
        self._log_file_path = self._logs_dir / "runner.log"

        self._maintainer = maintainer_cls()

        self._executor = executor_cls(
            workspace=self.EXECUTOR_WORKSPACE,
            maintainer=self._maintainer,
            pipeline=pipeline,
            devices=devices,
        )

        signal.signal(signal.SIGINT, self._catch)

        self._logs_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """
        Start runner

        Returns:
            None
        """
        self._setup_logger()

        task = self._preparer.exec(
            workspace=self.EXECUTOR_WORKSPACE,
            config=self._config,
            pipeline=self._pipeline,
            logs_dir=self._logs_dir,
            maintainer=self._maintainer,
            triton=Triton(),
        )

        results = []
        try:
            for result in self._executor.start(task):
                results.append(result)
        except RunnerException as e:
            LOGGER.error(f"Error running task: {str(e)}")
        finally:
            self._executor.stop()
            self._finalizer.exec(workspace=self.EXECUTOR_WORKSPACE, task=task, results=results)

    def _catch(self, signum, frame):
        """
        SIGINT catcher. Stops executor on any sigterm.

        Args:
            signum: signal id
            frame: signal frame
        """
        self._executor.stop()

        sys.exit(0)

    def _setup_logger(self) -> None:
        """
        Add file handle for logger

        Returns:
            None
        """
        file = logging.FileHandler(self._log_file_path)
        formatter = logging.Formatter(log_format)
        file.setFormatter(formatter)

        LOGGER.addHandler(file)
        LOGGER.setLevel(level=self._log_level)
        LOGGER.initialize(file_path=self._log_file_path)
