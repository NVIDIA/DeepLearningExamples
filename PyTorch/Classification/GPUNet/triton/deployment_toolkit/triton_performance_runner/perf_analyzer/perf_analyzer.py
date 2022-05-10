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
import pathlib
from subprocess import PIPE, CalledProcessError, Popen

# method from PEP-366 to support relative import in executed modules
from typing import List, Optional

if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .exceptions import PerfAnalyzerException

MAX_INTERVAL_CHANGES = 10
COUNT_INTERVAL_DELTA = 50
TIME_INTERVAL_DELTA = 2000

LOGGER = logging.getLogger(__name__)


class PerfAnalyzer:
    """
    This class provides an interface for running workloads
    with perf_analyzer.
    """

    def __init__(self, config, timeout: Optional[int]):
        """
        Parameters
        ----------
        config : PerfAnalyzerConfig
            keys are names of arguments to perf_analyzer,
            values are their values.
        """
        self.bin_path = "perf_analyzer"
        self._config = config
        self._output = ""
        self._timeout = timeout

    def run(self):
        """
        Runs the perf analyzer with the
        initialized configuration

        Returns
        -------
        List of Records
            List of the metrics obtained from this
            run of perf_analyzer

        Raises
        ------
        PerfAnalyzerException
            If subprocess throws CalledProcessError
        """
        self._output = ""

        for _ in range(MAX_INTERVAL_CHANGES):
            command = [self.bin_path]
            command += self._config.to_cli_string().replace("=", " ").split()

            LOGGER.debug(f"Perf Analyze command: {command}")
            if not self._timeout:
                LOGGER.debug("Perf Analyze command timeout not set")
            else:
                LOGGER.debug(f"Perf Analyze command timeout: {self._timeout} [s]")

            try:
                self._run_with_stream(command=command)
                return
            except CalledProcessError as e:
                if self._failed_with_measurement_inverval(e.output):
                    if self._config["measurement-mode"] is None or self._config["measurement-mode"] == "count_windows":
                        self._increase_request_count()
                    else:
                        self._increase_time_interval()
                else:
                    raise PerfAnalyzerException(
                        f"Running perf_analyzer with {e.cmd} failed with" f" exit status {e.returncode} : {e.output}"
                    )

        raise PerfAnalyzerException(f"Ran perf_analyzer {MAX_INTERVAL_CHANGES} times, but no valid requests recorded.")

    def output(self):
        """
        Returns
        -------
        The stdout output of the
        last perf_analyzer run
        """
        if self._output:
            return self._output
        raise PerfAnalyzerException("Attempted to get perf_analyzer output" "without calling run first.")

    def _run_with_stream(self, command: List[str]):
        commands_lst = []

        if self._timeout:
            commands_lst = ["timeout", str(self._timeout)]

        commands_lst.extend(command)
        LOGGER.debug(f"Run with stream: {commands_lst}")
        process = Popen(commands_lst, start_new_session=True, stdout=PIPE, encoding="utf-8")
        streamed_output = ""
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                streamed_output += output
                print(output.rstrip())

        self._output += streamed_output
        result = process.poll()
        LOGGER.debug(f"Perf Analyzer process exited with result: {result}")

        # WAR for Perf Analyzer exit code 0 when stabilization failed
        if result == 0 and self._failed_with_measurement_inverval(streamed_output):
            LOGGER.debug("Perf Analyzer finished with exit status 0, however measurement stabilization failed.")
            result = 1

        if result != 0:
            raise CalledProcessError(returncode=result, cmd=commands_lst, output=streamed_output)

    def _failed_with_measurement_inverval(self, output: str):
        checks = [
            output.find("Failed to obtain stable measurement"),
            output.find("Please use a larger time window"),
        ]
        result = any([status != -1 for status in checks])

        LOGGER.debug(f"Measurement stability message validation: {checks}. Result: {result}.")
        return result

    def _increase_request_count(self):
        self._config["measurement-request-count"] += COUNT_INTERVAL_DELTA
        LOGGER.debug(
            "perf_analyzer's measurement request count is too small, "
            f"increased to {self._config['measurement-request-count']}."
        )

    def _increase_time_interval(self):
        self._config["measurement-interval"] += TIME_INTERVAL_DELTA
        LOGGER.debug(
            "perf_analyzer's measurement window is too small, "
            f"increased to {self._config['measurement-interval']} ms."
        )
