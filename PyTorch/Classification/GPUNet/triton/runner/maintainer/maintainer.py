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
import abc
import pathlib
from typing import Any, Dict, List, Optional, Union

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .container import Container


class Maintainer(abc.ABC):
    @abc.abstractmethod
    def triton_container(
        self, command: str, image: str, devices: List, volumes: Dict, environment: Dict, log_file: Union[pathlib.Path, str]
    ) -> Container:
        """
        Return triton container

        Args:
            command: Triton Server command that has to be executed
            image: Container image
            devices: List of device ids which has to be available in container
            volumes: Volumes mapping
            environment: Environment variables set in container
            log_file: File path where server logs has to be saved

        Returns:
            Container object
        """
        pass

    @abc.abstractmethod
    def build_image(
        self,
        *,
        image_file_path: pathlib.Path,
        image_name: str,
        workdir_path: Optional[pathlib.Path] = None,
        build_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass
