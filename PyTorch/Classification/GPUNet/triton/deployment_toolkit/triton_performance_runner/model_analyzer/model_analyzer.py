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
import logging
import subprocess
from subprocess import CalledProcessError

from .exceptions import ModelAnalyzerException

SERVER_OUTPUT_TIMEOUT_SECS = 5
LOGGER = logging.getLogger(__name__)


class ModelAnalyzerMode:
    PROFILE = "profile"
    ANALYZE = "analyze"
    REPORT = "report"


class ModelAnalyzerReportMode:
    OFFLINE = "offline"
    ONLINE = "online"


class ModelAnalyzer:
    """
    Concrete Implementation of Model Analyzer interface that runs
    analyzer locally as as subprocess.
    """

    _analyzer_path = "model-analyzer"

    def __init__(self, config, timeout: int = None):
        """
        Parameters
        ----------
        config : AnalyzerConfig
            the config object containing arguments for this server instance
        """

        self._analyzer_process = None
        self._analyzer_config = config
        self._log = None
        self._timeout = timeout

    def run(self, mode: str, verbose: bool = False, quiet: bool = False, report_mode: str = None):
        """
        Starts the model analyzer locally
        """

        if self._analyzer_path:

            cmd = []
            if self._timeout:
                cmd = ["timeout", str(self._timeout)]

            cmd += [self._analyzer_path]
            if verbose:
                cmd += ["--verbose"]

            if quiet:
                cmd += ["--quiet"]

            if report_mode:
                cmd += ["-m"]
                cmd += [report_mode]

            cmd += [mode]
            cmd += self._analyzer_config.to_cli_string().split()

            LOGGER.debug(f"Model Analyze command: {cmd}")
            try:
                subprocess.run(cmd, check=True, start_new_session=True)

            except CalledProcessError as e:
                raise ModelAnalyzerException(
                    f"Running {self._analyzer_path} with {e.cmd} failed with"
                    f" exit status {e.returncode} : {e.output}"
                )
