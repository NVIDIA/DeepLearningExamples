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
import os
import pathlib

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .core import Command
from .exceptions import RunnerException
from .stages import Stage


class CommandsExporter:
    """
    Command exported to BASH scripts
    """

    def __init__(self, scripts_dir: pathlib.Path):
        """

        Args:
            scripts_dir: Paths where scripts should be stored
        """
        self._scripts_dir = scripts_dir

    def export(self, stage: Stage) -> Command:
        """
        Export stage commands to script and return new command to execute

        Args:
            stage: Stage object with commands

        Returns:
            Command object with script execution command
        """
        filename = self._get_filename(stage.label)
        file_path = self._scripts_dir / filename
        with open(file_path, "w+") as stagefile:
            stagefile.write("set -x\n")
            stagefile.write("set -e\n")
            stagefile.write("export PYTHONUNBUFFERED=1\n")
            stagefile.write("export PYTHONPATH=`pwd`\n")
            for command in stage.commands:
                stagefile.write(str(command))

        result = os.system(f'ex +"set syn=sh" +"norm gg=G" -cwq {file_path}')
        if result != 0:
            raise RunnerException(f"Failed running {filename} script formatting. Exit code {result}")

        command = Command(f"bash -xe {file_path.as_posix()}")
        return command

    def _get_filename(self, label: str):
        """
        Generate filename for script based on label

        Args:
            label: String with stage label

        Returns:
            String with script filename
        """
        filename = label.replace(" ", "_").lower()
        filename = f"{filename}.sh"

        return filename
