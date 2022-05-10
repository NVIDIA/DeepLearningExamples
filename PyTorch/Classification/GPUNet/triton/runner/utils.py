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
import os
import pathlib
import shutil
import subprocess
from enum import Enum
from typing import Any

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .core import Command
from .exceptions import RunnerException
from .logger import LOGGER


def format_env_key(s: str):
    """
    Format environmental variable key

    Args:
        s: String to format

    Returns:
        Upper cased string
    """
    return s.upper()


def format_env_value(value: Any) -> str:
    """
    Format environment variable value

    Args:
        value: value to be formatted

    Returns:
        Formatted value as a string
    """
    value = value if not isinstance(value, Enum) else value.value
    value = value if type(value) not in [list, tuple] else ",".join(map(str, value))
    value = str(value)
    return value


def get_result_path(result_path: str) -> str:
    """
    Map result path when different variants passed ex. with env variable in path

    Args:
        result_path: Path to result file

    Returns:
        str
    """
    for env_var, val in os.environ.items():
        result_path = result_path.replace(f"${{{env_var}}}", val)

    if result_path.startswith("/"):
        return result_path

    if result_path.startswith("./"):
        result_path = result_path[2:]

    return result_path


def clean_directory(directory: pathlib.Path) -> None:
    """
    Remove all files and directories from directory

    Args:
        directory: Path to directory which should be cleaned

    Returns:
        None
    """
    LOGGER.debug(f"Cleaning {directory.as_posix()}")
    if not directory.is_dir():
        LOGGER.warning(f"{directory.name} is not a directory.")
        return

    for item in os.listdir(directory):
        item_path = directory / item
        if item_path.is_dir():
            LOGGER.debug(f"Remove dir {item_path.as_posix()}")
            shutil.rmtree(item_path.as_posix())
        elif item_path.is_file():
            LOGGER.debug(f"Remove file: {item_path.as_posix()}")
            item_path.unlink()
        else:
            LOGGER.warning(f"Cannot remove item {item_path.name}. Not a file or directory.")


def exec_command(command: Command) -> None:
    """
    Execute command

    Args:
        command: Command to run
    """
    try:
        process = subprocess.Popen(
            [str(command)],
            shell=True,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
        )
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break

            if output:
                print(output.rstrip())
                LOGGER.write(output)

        result = process.poll()
        if result != 0:
            raise RunnerException(f"Command {command} failed with exit status: {result}")

    except subprocess.CalledProcessError as e:
        raise RunnerException(f"Running command {e.cmd} failed with exit status {e.returncode} : {e.output}")


def measurement_env_params(measurement):
    params = {}
    for key, value in measurement.__dict__.items():
        param = f"{measurement.__class__.__name__.upper()}_{key.upper()}"
        params[param] = " ".join(list(map(lambda val: str(val), value))) if isinstance(value, list) else int(value)

    return params


def offline_performance_configuration(steps, max_batch_size):
    step = int(max_batch_size) // steps
    batch_sizes = [step * idx for idx in range(1, steps + 1)]
    concurrency = [1]
    return batch_sizes, concurrency


def online_performance_configuration(steps, max_batch_size, number_of_model_instances):
    max_total_requests = 2 * int(max_batch_size) * int(number_of_model_instances)
    max_concurrency = min(128, max_total_requests)
    step = max(1, max_concurrency // steps)
    min_concurrency = step
    batch_sizes = [max(1, max_total_requests // max_concurrency)]
    concurrency = list(range(min_concurrency, max_concurrency + 1, step))

    return batch_sizes, concurrency
