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
from typing import Any, Dict, List, Optional, Union

import docker

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ...logger import LOGGER
from ..maintainer import Maintainer
from .container import DockerContainer
from .containers import TritonServerContainer


class DockerMaintainer(Maintainer):
    def triton_container(
        self, command: str, image: str, devices: List, volumes: Dict, environment: Dict, log_file: Union[pathlib.Path, str]
    ) -> DockerContainer:
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
            DockerContainer object
        """
        return TritonServerContainer(
            name="triton-server",
            command=command,
            image=image,
            devices=devices,
            volumes=volumes,
            environment=environment,
            log_file=log_file,
        )

    def build_image(
        self,
        *,
        image_file_path: pathlib.Path,
        image_name: str,
        workdir_path: Optional[pathlib.Path] = None,
        build_args: Optional[Dict[str, Any]] = None,
    ) -> None:

        workdir_path = workdir_path or image_file_path.parent
        build_args = build_args or {}
        LOGGER.info(f"Building {image_name} docker image.")
        LOGGER.debug(f"    Using workdir: {workdir_path}")
        LOGGER.debug(f"    Dockerfile: {image_file_path}")
        LOGGER.debug(f"    Build args: {build_args}")
        build_logs = list()
        try:
            docker_client = docker.from_env()
            _, build_logs = docker_client.images.build(
                path=workdir_path.resolve().as_posix(),
                dockerfile=image_file_path.resolve().as_posix(),
                tag=image_name,
                buildargs=build_args,
                network_mode="host",
                rm=True,
            )
        except docker.errors.BuildError as e:
            build_logs = e.build_log
            raise e
        finally:
            for chunk in build_logs:
                log = chunk.get("stream")
                if log:
                    LOGGER.debug(log.rstrip())
