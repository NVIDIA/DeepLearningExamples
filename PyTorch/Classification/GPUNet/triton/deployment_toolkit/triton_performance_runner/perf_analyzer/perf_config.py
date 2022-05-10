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
from typing import Any

from .exceptions import PerfAnalyzerException


class PerfAnalyzerConfig:
    """
    A config class to set arguments to the perf_analyzer.
    An argument set to None will use the perf_analyzer's default.
    """

    perf_analyzer_args = [
        "async",
        "sync",
        "measurement-interval",
        "measurement-mode",
        "measurement-request-count",
        "concurrency-range",
        "request-rate-range",
        "request-distribution",
        "request-intervals",
        "binary-search",
        "num-of-sequence",
        "latency-threshold",
        "max-threads",
        "stability-percentage",
        "max-trials",
        "percentile",
        "input-data",
        "shared-memory",
        "output-shared-memory-size",
        "sequence-length",
        "string-length",
        "string-data",
    ]

    perf_analyzer_multiple_args = [
        "shape",
    ]

    input_to_options = [
        "model-name",
        "model-version",
        "batch-size",
        "url",
        "protocol",
        "latency-report-file",
        "streaming",
    ]

    input_to_verbose = ["verbose", "extra-verbose"]

    def __init__(self):
        """
        Construct a PerfAnalyzerConfig
        """

        self._args = {k: None for k in self.perf_analyzer_args}
        self._multiple_args = {k: [] for k in self.perf_analyzer_multiple_args}

        self._options = {
            "-m": None,
            "-x": None,
            "-b": None,
            "-u": None,
            "-i": None,
            "-f": None,
            "-H": None,
            "-c": None,
            "-t": None,
        }
        self._verbose = {"-v": None, "-v -v": None}

        self._input_to_options = {
            "model-name": "-m",
            "model-version": "-x",
            "batch-size": "-b",
            "url": "-u",
            "protocol": "-i",
            "latency-report-file": "-f",
            "streaming": "-H",
            "concurrency": "-c",
            "threads": "-t",
        }

        self._input_to_verbose = {"verbose": "-v", "extra-verbose": "-v -v"}

    @classmethod
    def allowed_keys(cls):
        """
        Returns
        -------
        list of str
            The keys that are allowed to be
            passed into perf_analyzer
        """

        return (
            list(cls.perf_analyzer_args)
            + list(cls.perf_analyzer_multiple_args)
            + list(cls.input_to_options)
            + list(cls.input_to_verbose)
        )

    def update_config(self, params=None):
        """
        Allows setting values from a
        params dict

        Parameters
        ----------
        params: dict
            keys are allowed args to perf_analyzer
        """

        if params:
            for key in params:
                self[key] = params[key]

    def to_cli_string(self):
        """
        Utility function to convert a config into a
        string of arguments to the perf_analyzer with CLI.

        Returns
        -------
        str
            cli command string consisting of all arguments
            to the perf_analyzer set in the config, without
            the executable name.
        """

        # single dashed options, then verbose flags, then main args
        args = [f"{k} {v}" for k, v in self._options.items() if v]
        args += [k for k, v in self._verbose.items() if v]
        args += [f"--{k}={v}" for k, v in self._args.items() if v]
        for k, v in self._multiple_args.items():
            for item in v:
                args.append(f"--{k}={item}")

        return " ".join(args)

    def __getitem__(self, key: str):
        """
        Gets an arguments value in config

        Parameters
        ----------
        key : str
            The name of the argument to the perf_analyzer

        Returns
        -------
            The value that the argument is set to in this config

        Raises
        ------
        TritonModelAnalyzerException
            If argument not found in the config
        """

        if key in self._args:
            return self._args[key]
        elif key in self._multiple_args:
            return self._multiple_args[key]
        elif key in self._input_to_options:
            return self._options[self._input_to_options[key]]
        elif key in self._input_to_verbose:
            return self._verbose[self._input_to_verbose[key]]
        else:
            raise PerfAnalyzerException(f"'{key}' Key not found in config")

    def __setitem__(self, key: str, value: Any):
        """
        Sets an arguments value in config
        after checking if defined/supported.

        Parameters
        ----------
        key : str
            The name of the argument to the perf_analyzer
        value : (any)
            The value to which the argument is being set

        Raises
        ------
        TritonModelAnalyzerException
            If key is unsupported or undefined in the
            config class
        """

        if key in self._args:
            self._args[key] = value
        elif key in self._multiple_args:
            self._multiple_args[key].append(value)
        elif key in self._input_to_options:
            self._options[self._input_to_options[key]] = value
        elif key in self._input_to_verbose:
            self._verbose[self._input_to_verbose[key]] = value
        else:
            raise PerfAnalyzerException(
                f"The argument '{key}' to the perf_analyzer " "is not supported by the model analyzer."
            )
