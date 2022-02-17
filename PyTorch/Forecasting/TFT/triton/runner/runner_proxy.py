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
import pathlib
from typing import List, Type

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .config import Config
from .executor import Executor
from .finalizer import Finalizer
from .maintainer import Maintainer
from .pipeline import Pipeline
from .preparer import Preparer
from .runner import Runner


class RunnerProxy:
    """
    Runner proxy to configure original runner
    """

    maintainer_cls: Type[Maintainer] = None
    executor_cls: Type[Executor] = None
    preparer_cls: Type[Preparer] = None
    finalizer_cls: Type[Finalizer] = None

    def __init__(self, config: Config, pipeline: Pipeline, devices: List[str]):
        """
        RunnerProxy constructor

        Args:
            config: Config object
            pipeline: Pipeline to evaluate
            devices: List of devices to use for tests
        """
        self._runner = Runner(
            config=config,
            pipeline=pipeline,
            devices=devices,
            maintainer_cls=self.maintainer_cls,
            executor_cls=self.executor_cls,
            preparer_cls=self.preparer_cls,
            finalizer_cls=self.finalizer_cls,
        )

    def start(self) -> None:
        """
        Runner interface
        """
        self._runner.start()
